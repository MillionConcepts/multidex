import statistics as stats
from typing import Tuple, List

import fs.path
import PIL
import pandas as pd
from PIL import Image
from django.db import models
from toolz import keyfilter

from plotter_utils import modeldict

# TODO: REWRITE THIS
# def filter_fields(model):
# return [
#     field.name
#     for field in model._meta.get_fields()
#     if (
#             field.name[0:-5] in model.filters.keys()
#             or field.name[0:-6] in model.filters.keys()
#     )
# ]


MSPEC_IMAGE_TYPES = [
    "righteye_roi_image_1",
    "righteye_roi_image_2",
    "righteye_roi_image_3",
    "righteye_roi_image_4",
    "righteye_roi_image_5",
    "righteye_roi_image_6",
    "righteye_roi_image_7",
    "righteye_roi_image_8",
    "lefteye_roi_image_1",
    "lefteye_roi_image_2",
    "lefteye_roi_image_3",
    "lefteye_roi_image_4",
    "lefteye_roi_image_5",
    "lefteye_roi_image_6",
    "lefteye_roi_image_7",
    "lefteye_roi_image_8",
    "righteye_rgb_image_1",
    "righteye_rgb_image_2",
    "righteye_rgb_image_3",
    "righteye_rgb_image_4",
    "lefteye_rgb_image_1",
    "lefteye_rgb_image_2",
    "lefteye_rgb_image_3",
    "lefteye_rgb_image_4"
]


class Observation(models.Model):
    """
    class representing a single observation consisting of one or more images
    of the same scene,
    possibly created from a multispectral series of images, 
    possibly along with named ROIs that correspond to reduced spectral data
    contained in Spectrum
    objects. ROIs as drawn on each image notionally correspond to the same
    physical locations.

    this is a parent class intended to be subclassed for individual
    instruments.
    """

    name = models.CharField("Name", max_length=100, db_index=True)

    def spectra(self):
        return self.spectra_set.all()


class Spectrum(models.Model):
    """
    class representing reduced spectral data derived from a named ROI on one
    or more images.

    this is a parent class intended to be subclassed for individual
    instruments.
    """

    # this is not intended to hold all transmission information about the
    # filters, unlike FilterSet from wwu_spec (or some version that may be
    # instantiated here later) it is primarily a container. so just: filter
    # names and canonical center values.

    filters = {}
    virtual_filters = {}
    virtual_filter_mapping = {}
    axis_value_properties = {}
    marker_value_properties = {}
    searchable_fields = []

    def all_filter_waves(self):
        return self.filters | self.virtual_filters

    def as_table(self) -> pd.DataFrame:
        """
        this will sometimes be faster and sometimes slower than just directly
        accessing database fields or using as_dict. unlike as_dict, this
        specifically leaves 'None' values in. i guess it's maybe intended as
        a slightly lower-level API element.
        """
        rows = []
        for filt, wave in self.filters.items():
            mean = getattr(self, filt)
            err = getattr(self, filt + "_err")
            rows.append([filt, wave, mean, err])
        return pd.DataFrame(rows)

    # TODO: optimization notes
    # for most of the below calculations, i think a dictionary and standard
    # library functions are faster than a dataframe or numpy array and vect
    # functions, unless we have spectra measured with > ~50-100 points. it's
    # possible that i'm making too many lookups, though, and this should all
    # be refactored to reduce that.
    # adding scaling and nearby filter averaging makes this worse, because you
    # have to redo the scale/average lookups for each axis as it's currently
    # written. This should probably be refactored to reduce that.
    # the best way to do this would be to populate a dataframe with the
    # filter values of everything in the queryset, and perform operations
    # on that.

    def filter_values(self, **scale_kwargs) -> dict:
        """
        dictionary of filter name: {wavelength, mean reflectance}. used to
        initialize most functions. to be replaced in some subclasses with
        whatever weird averaging and scaling rules might be desired
        """
        return {
            filt: {'wave': self.filters[filt], 'mean': getattr(self, filt)}
            for filt in self.filters
            if getattr(self, filt)
        }

    def intervening(
            self, wave_1: float, wave_2: float,
            **scale_kwargs) -> List[
        Tuple[int, float]
    ]:
        """
        wavelength, mean reflectance for all bands strictly between
        wave_1 & wave_2
        """
        return [
            (filt['wave'], filt['mean'])
            for filt in self.filter_values(**scale_kwargs).values()
            if (
                    max([wave_1, wave_2]) > filt['wave'] > min([wave_1, wave_2])
                    and filt['mean'] is not None # TODO: Cruft?
            )
        ]

    def band(self, wave_1: float, wave_2: float, **scale_kwargs) -> List[
        Tuple[int, float]
    ]:
        """
        wavelength, mean reflectance for all bands between and including
        wave_1 & wave_2 
        """
        return [
            (filt['wave'], filt['mean'])
            for filt in self.filter_values(**scale_kwargs).values()
            if (
                max([wave_1, wave_2]) >= filt['wave'] >= min([wave_1, wave_2])
                and filt['mean'] is not None  # TODO: Cruft?
            )
        ]

    def ref(self, filt_1: str, **scale_kwargs) -> float:
        """mean reflectance at filter given by filter_name"""
        return self.filter_values(**scale_kwargs)[filt_1]['mean']

    def band_avg(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
        """
        average of reflectance values at filt_1, filt_2, and all intervening
        bands. this currently double-counts measurements at matching
        wavelengths in cases where bands are not being virtually averaged.
        will cause issues if you ask for filters that aren't there when
        virtually averaging and also ask for things to be virtually averaged.
        """
        return stats.mean([
            wave_reflectance_tuple[1]
            for wave_reflectance_tuple in self.band(
                self.all_filter_waves()[filt_1], self.all_filter_waves()[filt_2], **scale_kwargs
            )])

    def band_max(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
        """
        max reflectance value between filt_1 and filt_2 (inclusive)
        """
        return max([
            wave_reflectance_tuple[1]
            for wave_reflectance_tuple in self.band(
                self.all_filter_waves()[filt_1], self.all_filter_waves()[filt_2], **scale_kwargs
            )])

    def band_min(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
        """
        min reflectance value between filt_1 and filt_2 (inclusive)
        """
        return min([
            wave_reflectance_tuple[1]
            for wave_reflectance_tuple in self.band(
                self.all_filter_waves()[filt_1], self.all_filter_waves()[filt_2], **scale_kwargs
            )])

    def ref_ratio(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
        """
        ratio of reflectance values at filt_1 & filt_2
        """
        filts = self.filter_values(**scale_kwargs)
        return filts[filt_1]['mean'] / filts[filt_2]['mean']

    def slope(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
        """
        where input strings are filter names defined in self.filters

        slope of line drawn between reflectance values at these two filters.
        passing filt_1 == filt_2 returns an error.
        do we allow 'backwards' lines? for now yes
        """
        if filt_1 == filt_2:
            raise ValueError(
                "slope between a wavelength and itself is undefined"
            )
        filts = self.filter_values(**scale_kwargs)
        ref_1 = filts[filt_1]['mean']
        ref_2 = filts[filt_2]['mean']
        wave_1 = filts[filt_1]['wave']
        wave_2 = filts[filt_2]['wave']
        return (ref_2 - ref_1) / (wave_2 - wave_1)

    def band_depth_custom(
        self,
        filt_left: str,
        filt_right: str,
        filt_middle: str,
        **scale_kwargs
    ) -> float:
        """
        simple band depth at filt_middle --
        filt_middle reflectance / reflectance of 'continuum'
        (straight line between filt_left and filt_right) at filt_middle.
        passing filt_left == filt_right or filt_middle not strictly between
        them returns an error

        do we allow 'backwards' lines? for now yes (so 'left' and 'right'
        are misnomers)
        """
        if len({filt_left, filt_middle, filt_right}) != 3:
            raise ValueError(
                "band depth between a wavelength and itself is undefined"
            )
        filts = self.filter_values(**scale_kwargs)
        wave_left = filts[filt_left]['wave']
        wave_middle = filts[filt_middle]['wave']
        wave_right = filts[filt_right]['wave']

        if not (
                max(wave_left, wave_right)
                > wave_middle
                > min(wave_left, wave_right)
        ):
            raise ValueError(
                "band depth can only be calculated at a band within the "
                "chosen range."
            )
        distance = wave_middle - wave_left
        slope = self.slope(filt_left, filt_right, **scale_kwargs)
        continuum_ref = filts[filt_left]['mean'] + slope * distance
        return filts[filt_middle]['mean'] / continuum_ref

    def band_depth_min(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
        """
        simple band depth at local minimum --
        local minimum reflectance / reflectance of 'continuum'
        (straight line between filt_1 and filt_2) at that wavelength.
        band depth between adjacent points is defined as 1. 
        passing filt_1 == filt_2 returns an error.

        do we allow 'backwards' lines? for now yes
        """
        if filt_1 == filt_2:
            raise ValueError(
                "band depth between a wavelength and itself is undefined"
            )
        filts = self.filter_values(**scale_kwargs)
        wave_1 = filts[filt_1]['wave']
        wave_2 = filts[filt_2]['wave']

        intervening = self.intervening(wave_1, wave_2, **scale_kwargs)
        if not intervening:
            return 1

        min_ref = min([filt[1] for filt in intervening])
        min_wave = [
            (ix, wave)
            for ix, wave in enumerate([filt[0] for filt in intervening])
            if intervening[ix][1] == min_ref
        ][0]

        distance = min_wave[0] - wave_1
        slope = self.slope(filt_1, filt_2, **scale_kwargs)
        continuum_ref = filts[filt_1]['mean'] + slope * distance
        return min_ref / continuum_ref


    def clean(self):
        # TODO: likely remove this
        # for virtual_filter, corresponding_filters in \
        #         self.virtual_filter_mapping.items():
        #     filter_values = [
        #         getattr(self, filt) for filt in corresponding_filters
        #     ]
        #     if any([value is None for value in filter_values]):
        #         continue
        #     setattr(
        #         self,
        #         virtual_filter,
        #         np.mean(np.array(filter_values))
        #     )
        super(Spectrum, self).clean()


class MObs(Observation):
    """Observation subclass for MASTCAM"""

    sol = models.IntegerField("Sol", db_index=True)
    # note this is ltst for first frame of sequence (usually left eye 'clear')
    ltst = models.TimeField("Local True Solar Time", blank=True, null=True,
                            db_index=True)
    # not sure what this actually is. format is of sequence
    # number in PDS header, but _value_ corresponds to request id in the PDS
    # header
    seq_id = models.CharField("sequence id, e.g. 'mcam00001'", max_length=20,
                              db_index=True)
    rover_elevation = models.FloatField(
        "Rover Elevation", blank=True, null=True, db_index=True
    )
    target_elevation = models.FloatField(
        "Target Elevation", null=True, db_index=True
    )
    tau_interpolated = models.FloatField("Interpolated Tau", blank=True,
                                         null=True, db_index=True)
    # this field is duplicated in the original data set. assuming for now
    # that this is a mistake.
    focal_distance = models.FloatField(
        "Focal Distance", blank=True, null=True, db_index=True
    )
    incidence_angle = models.FloatField(
        "Incidence Angle", blank=True, null=True, db_index=True
    )
    emission_angle = models.FloatField(
        "Emission Angle", blank=True, null=True, db_index=True
    )
    l_s = models.FloatField(
        "Solar Longitude", blank=True, null=True, db_index=True
    )
    # i think an arbitrary number for the site. part of the ROVER_NAV_FRAME
    # coordinate system
    site = models.IntegerField("Site", blank=True, null=True, db_index=True)
    # similar
    drive = models.IntegerField("Drive", blank=True, null=True, db_index=True)

    # presumably planetographic lat/lon
    # not in the image labels in PDS
    lat = models.FloatField("Latitude", blank=True, null=True, db_index=True)
    lon = models.FloatField("Longitude", blank=True, null=True, db_index=True)

    # don't know what this is
    traverse = models.FloatField("Traverse", blank=True, null=True,
                                 db_index=True)

    filename = models.CharField("Archive CSV File", max_length=30,
                                db_index=True)

    # sometimes multiple images are produced for a single observation.
    # note that in some cases, duplicate left-eye images are produced
    # for multiple non-duplicate right-eye images in order to draw ROIs
    # corresponding to all the right-eye images and the narrower
    # field of view of the right eye (not being able to just draw all of
    # the ROIs from all the right-eye images on a single left-eye image
    # is presumably due to limitations in MERTOOLS).

    righteye_roi_image_1 = models.CharField(
        "path to first false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_1 = models.CharField(
        "path to first false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_2 = models.CharField(
        "path to second false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_2 = models.CharField(
        "path to second false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_3 = models.CharField(
        "path to third false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_3 = models.CharField(
        "path to third false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_4 = models.CharField(
        "path to fourth false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_4 = models.CharField(
        "path to fourth false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_5 = models.CharField(
        "path to fifth false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_5 = models.CharField(
        "path to fifth false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_6 = models.CharField(
        "path to sixth false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_6 = models.CharField(
        "path to sixth false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_7 = models.CharField(
        "path to seventh false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_7 = models.CharField(
        "path to seventh false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_roi_image_8 = models.CharField(
        "path to eighth false-color roi right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_roi_image_8 = models.CharField(
        "path to eighth false-color roi left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_rgb_image_1 = models.CharField(
        "path to first rgb right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_rgb_image_1 = models.CharField(
        "path to first rgb left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_rgb_image_2 = models.CharField(
        "path to second rgb right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_rgb_image_2 = models.CharField(
        "path to second rgb left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_rgb_image_3 = models.CharField(
        "path to third rgb right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_rgb_image_3 = models.CharField(
        "path to third rgb left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    righteye_rgb_image_4 = models.CharField(
        "path to fourth rgb right-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    lefteye_rgb_image_4 = models.CharField(
        "path to fourth rgb left-eye image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )

    def image_files(self):
        filedict = {
            image_type: getattr(self, image_type)
            for image_type in MSPEC_IMAGE_TYPES
            if getattr(self, image_type)
        }
        return filedict

    def __str__(self):
        # noinspection PyTypeChecker
        return "sol" + str(self.sol) + "_" + self.seq_id


class MSpec(Spectrum):
    """Spectrum subclass for MASTCAM"""

    # feature type (rock, dust, etc) (not apparently exported with Tina's
    # sample data)
    feature_type = models.CharField(
        "Feature Type", blank=True, max_length=50, db_index=True
    )

    # ##########################################################
    # ## relationships with parent observation and images
    # ##########################################################

    observation_class = MObs
    observation = models.ForeignKey(
        observation_class,
        on_delete=models.PROTECT,
        related_name="spectra_set",
        blank=True,
    )

    # color of associated ROI
    color = models.CharField(
        "ROI Color", blank=True, max_length=20, db_index=True
    )

    # for some but not all observations, multiple images are produced.
    # this field indicates which image the ROI is drawn on
    image_number = models.IntegerField(
        "image number", default=1, db_index=True
    )

    feature = models.CharField(
        "feature category", default=None, blank=True,
        null=True, db_index=True, max_length=45
    )

    # ############################################
    # ## lithological information -- relevant only to rocks ###
    # #########################################################

    float = models.BooleanField(
        "floating vs. in-place", blank=True, null=True, db_index=True
    )

    # large-to-small taxonomic categories for rock clusters
    formation = models.CharField(
        "Formation", blank=True, null=True, max_length=50, db_index=True
    )
    member = models.CharField(
        "Member", blank=True, null=True, max_length=50, db_index=True
    )

    filename = models.CharField(
        "Name of archive CSV file", max_length=50, db_index=True
    )

    # ## end lithological ###

    notes = models.CharField(
        "Notes", blank=True, max_length=100, db_index=True
    )

    # #############################################
    # ########## explicit fields for reflectance values at each filter
    # #############################################

    # i'm pretty sure multiple fields is better than a stringified dict
    # it permits SQL queries on spectra
    # and doesn't require thousands of fields b/c not high-res spectra

    # this is _extremely_ wordy. there may be an equally-safe
    # way to do this by iterating over filter dictionary,
    # but I am nervous about it because every change can create SQL migrations,
    # and the question of when properties are set is dicey.

    l0b = models.FloatField("l0 (Blue Bayer) mean", blank=True, null=True,
                            db_index=True)
    l0b_err = models.FloatField("l0 (Blue Bayer) err", blank=True, null=True,
                                db_index=True)
    l0g = models.FloatField("l0 (Green Bayer) mean", blank=True, null=True,
                            db_index=True)
    l0g_err = models.FloatField(
        "l0 (Green Bayer) err", blank=True, null=True, db_index=True
    )
    l0r = models.FloatField(
        "l0 (Red Bayer) mean",
        blank=True,
        null=True,
        db_index=True
    )
    l0r_err = models.FloatField(
        "l0 (Red Bayer) err",
        blank=True,
        null=True,
        db_index=True
    )
    r0b = models.FloatField(
        "r0 (Blue Bayer) mean",
        blank=True,
        null=True,
        db_index=True
    )
    r0b_err = models.FloatField(
        "r0 (Blue Bayer) err",
        blank=True,
        null=True,
        db_index=True
    )
    r0g = models.FloatField(
        "r0 (Green Bayer) mean",
        blank=True,
        null=True,
        db_index=True
    )
    r0g_err = models.FloatField(
        "r0 (Green Bayer) err",
        blank=True,
        null=True,
        db_index=True
    )
    r0r = models.FloatField(
        "r0 (Red Bayer) mean",
        blank=True,
        null=True,
        db_index=True
    )
    r0r_err = models.FloatField(
        "r0 (Red Bayer) err",
        blank=True,
        null=True,
        db_index=True
    )

    # real filters

    r1 = models.FloatField("r1 mean", blank=True, null=True, db_index=True)
    r1_err = models.FloatField("r1 err", blank=True, null=True, db_index=True)
    l1 = models.FloatField("l1 mean", blank=True, null=True, db_index=True)
    l1_err = models.FloatField("l1 err", blank=True, null=True, db_index=True)
    r2 = models.FloatField("r2 mean", blank=True, null=True, db_index=True)
    r2_err = models.FloatField("r2 err", blank=True, null=True, db_index=True)
    l2 = models.FloatField("l2 mean", blank=True, null=True, db_index=True)
    l2_err = models.FloatField("l2 err", blank=True, null=True, db_index=True)
    r3 = models.FloatField("r3 mean", blank=True, null=True, db_index=True)
    r3_err = models.FloatField("r3 err", blank=True, null=True, db_index=True)
    l3 = models.FloatField("l3 mean", blank=True, null=True, db_index=True)
    l3_err = models.FloatField("l3 err", blank=True, null=True, db_index=True)
    r4 = models.FloatField("r4 mean", blank=True, null=True, db_index=True)
    r4_err = models.FloatField("r4 err", blank=True, null=True, db_index=True)
    l4 = models.FloatField("l4 mean", blank=True, null=True, db_index=True)
    l4_err = models.FloatField("l4 err", blank=True, null=True, db_index=True)
    r5 = models.FloatField("r5 mean", blank=True, null=True, db_index=True)
    r5_err = models.FloatField("r5 err", blank=True, null=True, db_index=True)
    l5 = models.FloatField("l5 mean", blank=True, null=True, db_index=True)
    l5_err = models.FloatField("l5 err", blank=True, null=True, db_index=True)
    r6 = models.FloatField("r6 mean", blank=True, null=True, db_index=True)
    r6_err = models.FloatField("r6 err", blank=True, null=True, db_index=True)
    l6 = models.FloatField("l6 mean", blank=True, null=True, db_index=True)
    l6_err = models.FloatField("l6 err", blank=True, null=True, db_index=True)

    # virtual filters
    #
    # TODO: I think it's better to calculate virtual filters when samples
    #  enter the
    #     database so we don't do a bunch of additional lookups and
    #     computations
    #     for what are essentially static properties every time we look at a
    #     sample in an 'averaged' view, but this requires ongoing assessment
    #     -- Task IDs R0001, C0001, C0002

    # at least for MASTCAM, the set of virtual filters === the set of pairs of
    # real filters with nominal band center differences <= 5 nm. We would,
    # however,
    # prefer to define them explicitly rather than generate them
    # programmatically
    # applying this rule, for a variety of reasons.
    # the virtual mean reflectance in an ROI for a virtual filter is the
    # arithmetic
    # mean of the mean reflectance values in that ROI for the two real filters
    # in its associated pair.
    # the nominal band center of a virtual filter is the arithmetic mean of the
    # nominal band centers of the two real filters in its associated pair.
    # TODO: 1. is that rule about nominal band center correct?
    #       2. what is the stdev / variance? are we pretending this is shot
    #       noise,
    #          independent between the eyes, so just a sum of Poisson
    #          distributions?
    #     --Task IDs C0001, C0002

    # mappings from filter name to nominal band centers, in nm

    filters = {
        "l2": 445,
        "r2": 447,
        "l0b": 482,
        "r0b": 482,
        "l1": 527,
        "r1": 527,
        "l0g": 554,
        "r0g": 554,
        "l0r": 640,
        "r0r": 640,
        "l4": 676,
        "l3": 751,
        "r3": 805,
        "l5": 867,
        "r4": 908,
        "r5": 937,
        "l6": 1012,
        "r6": 1013,
    }

    virtual_filters = {
        "l2_r2": 446,
        "l0b_r0b": 482,
        "l1_r1": 527,
        "l0g_r0g": 554,
        "l0r_r0r": 640,
        "l6_r6": 1013
    }

    # which real filters do virtual filters correspond to?
    virtual_filter_mapping = {
        "l2_r2": ("l2", "r2"),
        "l0b_r0b": ("l0b", "r0b"),
        "l1_r1": ("l1", "r1"),
        "l0g_r0g": ("l0g", "r0g"),
        "l0r_r0r": ("l0r", "r0r"),
        "l6_r6": ("l6", "r6")
    }

    axis_value_properties = [
        {
            "label": "band value",
            "value": "ref",
            "type": "method",
            "arity": 1,
            "value_type": "quant"
        },
        {
            "label": "band average",
            "value": "band_avg",
            "type": "method",
            "arity": 2,
            "value_type": "quant"
        },
        {
            "label": "band maximum",
            "value": "band_max",
            "type": "method",
            "arity": 2, "value_type": "quant"
        },
        {
            "label": "band minimum",
            "value": "band_min",
            "type": "method",
            "arity": 2, "value_type": "quant"
        },
        {
            "label": "ratio",
            "value": "ref_ratio",
            "type": "method",
            "arity": 2, "value_type": "quant"
        },
        {
            "label": "band depth at middle filter",
            "value": "band_depth_custom",
            "type": "method",
            "arity": 3, "value_type": "quant"
        },
        {
            "label": "band depth at band minimum",
            "value": "band_depth_min",
            "type": "method",
            "arity": 2, "value_type": "quant"
        },

        {"label": "sol", "value": "sol", "type": "parent_property",
         "value_type": "quant"},
        {
            "label": "target elevation",
            "value": "target_elevation",
            "type": "parent_property", "value_type": "quant"
        },
        {
            "label": "local true solar time",
            "value": "ltst",
            "type": "parent_property", "value_type": "quant"
        },
    ]

    searchable_fields = [
        {"label": "formation", "value": "formation", "type": "self_property",
         "value_type": "qual"},
        {"label": "member", "value": "member", "type": "self_property",
         "value_type": "qual"},
        {"label": "sol", "value": "sol", "type": "parent_property",
         "value_type": "quant"},
        {"label": "color", "value": "color", "type": "self_property",
         "value_type": "qual"},
        # need to use dateutil.parser.parse or similar if you're going to
        # have this
        # {
        #     "label": "ltst",
        #     "type": "parent_property",
        #     "value_type": "quant",
        # },
        {"label": "seq_id", "value": "seq_id", "type": "parent_property",
         "value_type": "qual"},
        {"label": "tau_interpolated", "value": "tau_interpolated",
         "type": "parent_property", "value_type": "quant"},
    ]

    marker_value_properties = axis_value_properties + searchable_fields

    def filter_values(self, scale_to = None, average_filters = False):
        """
        scale_to: None or tuple of (lefteye filter name, righteye filter name)
        """
        values = {}
        lefteye_scale = 1
        righteye_scale = 1
        if scale_to is not None:
            scales = (getattr(self, scale_to[0]), getattr(self, scale_to[1]))
            # don't scale eyes to a value that doesn't exist
            if all(scales):
                filter_mean = stats.mean(scales)
                lefteye_scale = filter_mean / scales[0]
                righteye_scale = filter_mean / scales[1]
        real_filters_to_use = list(self.filters.keys())
        if average_filters is True:
            for virtual_filter, comps in self.virtual_filter_mapping.items():
                # do not attempt to average filters if both filters of
                # a pair are not present
                if not all([getattr(self, comp) for comp in comps]):
                    continue
                [real_filters_to_use.remove(comp) for comp in comps]
                values[virtual_filter] = {
                    'wave': self.virtual_filters[virtual_filter],
                    'mean': stats.mean((
                        getattr(self, comps[0]) * lefteye_scale,
                        getattr(self, comps[1]) * righteye_scale
                    ))
                }
        for real_filter in real_filters_to_use:
            mean_value = getattr(self, real_filter)
            if mean_value is None:
                continue
            if real_filter.startswith('r'):
                eye_scale = righteye_scale
            else:
                eye_scale = lefteye_scale
            values[real_filter] = {
                'wave': self.filters[real_filter],
                'mean': getattr(self, real_filter) * eye_scale
            }
        return dict(sorted(values.items(), key=lambda item: item[1]['wave']))

    def image_files(self) -> dict:
        filedict = {
            image_type: getattr(self.observation, image_type)
            for image_type in MSPEC_IMAGE_TYPES
            if (
                    str(self.image_number) in image_type
                    and getattr(self.observation, image_type) is not None
            )
        }
        return filedict

    def overlay_browse_file_info(self, image_directory: str) -> dict:
        """
        directory containing browse image files ->
        {'right_file':string, 'left_file': 'left':PIL.Image}
        browse versions of left- and right-eye false-color images
        containing the region of interest from which the spectrum
        in MSpec was drawn
        """
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            for eye in ["left", "right"]:
                if eye + "eye_roi" in image_type:
                    browse_filename = fs.path.splitext(filename)[
                                          0] + '_browse.jpg'
                    images[eye + "_file"] = browse_filename
                    with PIL.Image.open(
                            fs.path.join(image_directory, browse_filename)
                    ) as image:
                        images[eye + "_size"] = image.size
        return images

    # colors corresponding to ROIs drawn on false-color images by MASTCAM team.
    # these are all somewhat uncertain, as they're based on color picker
    # results
    # from compressed images (compressed _after_ the polygons were drawn). 
    # it would be better to have official documentation and uncompressed
    # images!
    # TODO: replace this with 'from marslab.compatibility import
    #  MERSPECT_COLOR_MAPPINGS'
    color_mappings = {
        "light green": "#80ff00",
        "red": "#dc133d",
        "dark red": "#7e0003",
        "dark blue": "#010080",
        "light blue": "#0000fe",
        "light purple": "#ff00fe",
        "dark purple": "#81007f",
        "yellow": "#ffff00",
        "teal": "#008083",
        "dark green": "#138013",
        "sienna": "#a1502e",
        "light cyan": "#00ffff",
        # following are somewhat speculative
        "pink": "#ddc39f",
        "goldenrod": "#fec069",
        "bright red": "#d60702"
    }

    def roi_hex_code(self) -> str:
        return self.color_mappings[self.color]

    def metadata_dict(self) -> dict:
        """
        placeholder? metadata-summarizing function. for printing etc.
        """
        spec_dict = modeldict(self)
        # noinspection PyTypeChecker
        obs_dict = modeldict(self.observation)
        fields_to_print = [
            'color',
            'seq_id',
            'sol',
            'ltst',
            'feature',
            'formation',
            'member',
            'float',
            'lat',
            'lon',
            'site',
        ]
        return keyfilter(
            lambda x: x in fields_to_print, spec_dict | obs_dict
        )
