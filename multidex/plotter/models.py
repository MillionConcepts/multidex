from collections.abc import Sequence
from typing import Optional

import fs.path
import PIL
import pandas as pd
from PIL import Image
from django.db import models
from toolz import keyfilter

from marslab.compatibility import (
    MERSPECT_COLOR_MAPPINGS,
    MERSPECT_M20_COLOR_MAPPINGS,
    DERIVED_CAM_DICT,
    polish_xcam_spectrum,
)
from plotter_utils import modeldict

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
    "lefteye_rgb_image_4",
]

ZSPEC_IMAGE_TYPES = [
    "rgb_image",
    "enhanced_image"
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
    # filters. it is primarily a container.

    filters = {}
    virtual_filters = {}
    virtual_filter_mapping = {}
    axis_value_properties = []
    accessible_properties = []
    graphable_properties = []
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


class MObs(Observation):
    """Observation subclass for MASTCAM"""

    sol = models.IntegerField("Sol", db_index=True)
    # note this is ltst for first frame of sequence (usually left eye 'clear')
    ltst = models.TimeField(
        "Local True Solar Time", blank=True, null=True, db_index=True
    )
    # not sure what this actually is. format is of sequence
    # number in PDS header, but _value_ corresponds to request id in the PDS
    # header
    seq_id = models.CharField(
        "sequence id, e.g. 'mcam00001'", max_length=20, db_index=True
    )
    rover_elevation = models.FloatField(
        "Rover Elevation", blank=True, null=True, db_index=True
    )
    target_elevation = models.FloatField(
        "Target Elevation", null=True, db_index=True
    )
    tau = models.FloatField("Tau", blank=True, null=True, db_index=True)
    focal_distance = models.FloatField(
        "Focal Distance", blank=True, null=True, db_index=True
    )
    incidence_angle = models.FloatField(
        "Incidence Angle", blank=True, null=True, db_index=True
    )
    emission_angle = models.FloatField(
        "Emission Angle", blank=True, null=True, db_index=True
    )
    phase_angle = models.FloatField(
        "Phase Angle", blank=True, null=True, db_index=True
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

    odometry = models.FloatField(
        "Odometry", blank=True, null=True, db_index=True
    )

    filename = models.CharField(
        "Archive CSV File", max_length=30, db_index=True
    )

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


class ZObs(Observation):
    """
    Observation subclass for ZCAM
    TODO: this may or may not be staying as-is. This may be removed to
     'flatten' data and metadata together for more flexible definition of
      instrument characteristics.
    currently this should be understood not to comprise a full 'observation'
    in the M20 mission sense, but rather a single 'pointing' within an 'observation',
    uniquely defined by a combination of sol, seq id, commanded instrument az,
    commanded instrument el.
    """

    sol = models.IntegerField("Sol", db_index=True)
    ltst = models.TimeField(
        "Local True Solar Time", blank=True, null=True, db_index=True
    )
    # TODO: this is the temporary replacement for being able to uniquely
    #  identify pointings by means of AZ / EL -- fully manual
    pointing_index = models.IntegerField(
        "Pointing Index", default=1, db_index=True
    )
    sclk = models.IntegerField(
        "Spacecraft Clock", blank=True, null=True, db_index=True
    )
    seq_id = models.CharField(
        "sequence id", max_length=20, db_index=True
    )
    rover_elevation = models.FloatField(
        "Rover Elevation", blank=True, null=True, db_index=True
    )
    target_elevation = models.FloatField(
        "Target Elevation", null=True, db_index=True
    )
    tau = models.FloatField("Tau", blank=True, null=True, db_index=True)
    focal_distance = models.FloatField(
        "Focal Distance", blank=True, null=True, db_index=True
    )
    incidence_angle = models.FloatField(
        "Incidence Angle", blank=True, null=True, db_index=True
    )
    emission_angle = models.FloatField(
        "Emission Angle", blank=True, null=True, db_index=True
    )
    phase_angle = models.FloatField(
        "Phase Angle", blank=True, null=True, db_index=True
    )
    l_s = models.FloatField(
        "Solar Longitude", blank=True, null=True, db_index=True
    )

    site = models.IntegerField("Site", blank=True, null=True, db_index=True)
    drive = models.IntegerField("Drive", blank=True, null=True, db_index=True)

    lat = models.FloatField("Latitude", blank=True, null=True, db_index=True)
    lon = models.FloatField("Longitude", blank=True, null=True, db_index=True)

    odometry = models.FloatField(
        "Odometry", blank=True, null=True, db_index=True
    )
    zoom = models.CharField(
        "Focal Length", max_length=12, blank=True, null=True, db_index=True
    )

    filename = models.CharField(
        "Archive CSV File", max_length=30, db_index=True
    )

    # we have no idea what the actual image rules are going to end
    # up looking like for this. right now we're just enhanced color
    # and DCS images in here.

    rgb_image = models.CharField(
        "path to rgb image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )
    enhanced_image = models.CharField(
        "path to enhanced-color image",
        blank=True,
        null=True,
        max_length=100,
        db_index=True,
    )

    def image_files(self):
        filedict = {
            image_type: getattr(self, image_type)
            for image_type in ZSPEC_IMAGE_TYPES
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
        blank=True
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
        "feature category",
        default=None,
        blank=True,
        null=True,
        db_index=True,
        max_length=45,
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

    l0b = models.FloatField(
        "l0 (Blue Bayer) mean", blank=True, null=True, db_index=True
    )
    l0b_err = models.FloatField(
        "l0 (Blue Bayer) err", blank=True, null=True, db_index=True
    )
    l0g = models.FloatField(
        "l0 (Green Bayer) mean", blank=True, null=True, db_index=True
    )
    l0g_err = models.FloatField(
        "l0 (Green Bayer) err", blank=True, null=True, db_index=True
    )
    l0r = models.FloatField(
        "l0 (Red Bayer) mean", blank=True, null=True, db_index=True
    )
    l0r_err = models.FloatField(
        "l0 (Red Bayer) err", blank=True, null=True, db_index=True
    )
    r0b = models.FloatField(
        "r0 (Blue Bayer) mean", blank=True, null=True, db_index=True
    )
    r0b_err = models.FloatField(
        "r0 (Blue Bayer) err", blank=True, null=True, db_index=True
    )
    r0g = models.FloatField(
        "r0 (Green Bayer) mean", blank=True, null=True, db_index=True
    )
    r0g_err = models.FloatField(
        "r0 (Green Bayer) err", blank=True, null=True, db_index=True
    )
    r0r = models.FloatField(
        "r0 (Red Bayer) mean", blank=True, null=True, db_index=True
    )
    r0r_err = models.FloatField(
        "r0 (Red Bayer) err", blank=True, null=True, db_index=True
    )
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

    # mappings from filter name to nominal band centers, in nm
    filters = DERIVED_CAM_DICT["MCAM"]["filters"]
    virtual_filters = DERIVED_CAM_DICT["MCAM"]["virtual_filters"]
    # which real filters do virtual filters correspond to?
    virtual_filter_mapping = DERIVED_CAM_DICT["MCAM"]["virtual_filter_mapping"]
    # if we're only giving options for averaged filters,
    # what is the canonical list?
    canonical_averaged_filters = DERIVED_CAM_DICT["MCAM"][
        "canonical_averaged_filters"
    ]
    accessible_properties = [
        {
            "label": "band value",
            "value": "ref",
            "type": "method",
            "arity": 1,
            "value_type": "quant",
        },
        {
            "label": "band slope",
            "value": "slope",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band average",
            "value": "band_avg",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band maximum",
            "value": "band_max",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band minimum",
            "value": "band_min",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "ratio",
            "value": "ratio",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band depth (middle)",
            "value": "band_depth_custom",
            "type": "method",
            "arity": 3,
            "value_type": "quant",
        },
        {
            "label": "band depth (minimum)",
            "value": "band_depth_min",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "target_elevation",
            "value": "target_elevation",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "ltst",
            "value": "ltst",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "formation",
            "value": "formation",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "member",
            "value": "member",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "sol",
            "value": "sol",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "feature",
            "value": "feature",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "color",
            "value": "color",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "name",
            "value": "name",
            "type": "parent_property",
            "value_type": "qual",
        },
        {
            "label": "seq_id",
            "value": "seq_id",
            "type": "parent_property",
            "value_type": "qual",
        },
        {
            "label": "tau",
            "value": "tau",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "lat",
            "value": "lat",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "lon",
            "value": "lon",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "focal_distance",
            "value": "focal_distance",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "emission_angle",
            "value": "emission_angle",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "incidence_angle",
            "value": "incidence_angle",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "phase_angle",
            "value": "phase_angle",
            "type": "parent_property",
            "value_type": "quant",
        },
    ]

    graphable_properties = [
        prop
        for prop in accessible_properties
        if prop["value"] not in ("color", "seq_id", "name")
    ]
    searchable_fields = [
        prop
        for prop in accessible_properties
        if (prop["type"] != "method") and prop["value"] not in "ltst"
    ]

    def filter_values(
        self,
        scale_to: Optional[Sequence] = None,
        average_filters: bool = False,
    ) -> dict[str, dict]:
        """
        return dictionary of filter values, optionally scaled and merged
        according to MERSPECT-style rules
        scale_to: None or tuple of (lefteye filter name, righteye filter name)
        """
        spectrum = {
            filt: getattr(self, filt.lower()) for filt in self.filters.keys()
        }
        spectrum |= {
            filt + "_ERR": getattr(self, filt.lower() + "_err")
            for filt in self.filters.keys()
        }
        return polish_xcam_spectrum(
            spectrum=spectrum,
            cam_info=DERIVED_CAM_DICT["MCAM"],
            scale_to=scale_to,
            average_filters=average_filters,
        )

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
                    browse_filename = (
                        fs.path.splitext(filename)[0] + "_browse.jpg"
                    )
                    images[eye + "_file"] = browse_filename
                    with PIL.Image.open(
                        fs.path.join(image_directory, browse_filename)
                    ) as image:
                        images[eye + "_size"] = image.size
        return images

    # color corresponding to ROIs drawn on false-color images by MASTCAM team.
    def roi_hex_code(self) -> str:
        return MERSPECT_COLOR_MAPPINGS[self.color]

    def metadata_dict(self) -> dict:
        """
        metadata-summarizing function. could be made more efficient.
        """
        spec_dict = modeldict(self)
        # noinspection PyTypeChecker
        obs_dict = modeldict(self.observation)
        return keyfilter(
            lambda x: x
            in [field["value"] for field in self.searchable_fields],
            spec_dict | obs_dict,
        )


class ZSpec(Spectrum):
    """Spectrum subclass for ZCAM"""

    # feature type (rock, dust, etc) (not apparently exported with Tina's
    # sample data)
    feature_type = models.CharField(
        "Feature Type", blank=True, max_length=50, db_index=True
    )

    # ##########################################################
    # ## relationships with parent observation and images
    # ##########################################################

    observation_class = ZObs
    observation = models.ForeignKey(
        observation_class,
        on_delete=models.PROTECT,
        related_name="spectra_set",
        blank=True,
        null=True
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
        "feature category",
        default=None,
        blank=True,
        null=True,
        db_index=True,
        max_length=45,
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

    l0b = models.FloatField(
        "l0 (Blue Bayer) mean", blank=True, null=True, db_index=True
    )
    l0b_err = models.FloatField(
        "l0 (Blue Bayer) err", blank=True, null=True, db_index=True
    )
    l0g = models.FloatField(
        "l0 (Green Bayer) mean", blank=True, null=True, db_index=True
    )
    l0g_err = models.FloatField(
        "l0 (Green Bayer) err", blank=True, null=True, db_index=True
    )
    l0r = models.FloatField(
        "l0 (Red Bayer) mean", blank=True, null=True, db_index=True
    )
    l0r_err = models.FloatField(
        "l0 (Red Bayer) err", blank=True, null=True, db_index=True
    )
    r0b = models.FloatField(
        "r0 (Blue Bayer) mean", blank=True, null=True, db_index=True
    )
    r0b_err = models.FloatField(
        "r0 (Blue Bayer) err", blank=True, null=True, db_index=True
    )
    r0g = models.FloatField(
        "r0 (Green Bayer) mean", blank=True, null=True, db_index=True
    )
    r0g_err = models.FloatField(
        "r0 (Green Bayer) err", blank=True, null=True, db_index=True
    )
    r0r = models.FloatField(
        "r0 (Red Bayer) mean", blank=True, null=True, db_index=True
    )
    r0r_err = models.FloatField(
        "r0 (Red Bayer) err", blank=True, null=True, db_index=True
    )
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

    # mappings from filter name to nominal band centers, in nm
    filters = DERIVED_CAM_DICT["ZCAM"]["filters"]
    virtual_filters = DERIVED_CAM_DICT["ZCAM"]["virtual_filters"]
    # which real filters do virtual filters correspond to?
    virtual_filter_mapping = DERIVED_CAM_DICT["ZCAM"]["virtual_filter_mapping"]
    # if we're only giving options for averaged filters,
    # what is the canonical list?
    canonical_averaged_filters = DERIVED_CAM_DICT["ZCAM"][
        "canonical_averaged_filters"
    ]
    accessible_properties = [
        {
            "label": "band value",
            "value": "ref",
            "type": "method",
            "arity": 1,
            "value_type": "quant",
        },
        {
            "label": "band slope",
            "value": "slope",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band average",
            "value": "band_avg",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band maximum",
            "value": "band_max",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band minimum",
            "value": "band_min",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "ratio",
            "value": "ratio",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "band depth (middle)",
            "value": "band_depth_custom",
            "type": "method",
            "arity": 3,
            "value_type": "quant",
        },
        {
            "label": "band depth (minimum)",
            "value": "band_depth_min",
            "type": "method",
            "arity": 2,
            "value_type": "quant",
        },
        {
            "label": "target_elevation",
            "value": "target_elevation",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "ltst",
            "value": "ltst",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "sclk",
            "value": "sclk",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "zoom",
            "value": "zoom",
            "type": "parent_property",
            "value_type": "qual",
        },
        {
            "label": "formation",
            "value": "formation",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "member",
            "value": "member",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "sol",
            "value": "sol",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "feature",
            "value": "feature",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "color",
            "value": "color",
            "type": "self_property",
            "value_type": "qual",
        },
        {
            "label": "name",
            "value": "name",
            "type": "parent_property",
            "value_type": "qual",
        },
        {
            "label": "seq_id",
            "value": "seq_id",
            "type": "parent_property",
            "value_type": "qual",
        },
        {
            "label": "tau",
            "value": "tau",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "lat",
            "value": "lat",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "lon",
            "value": "lon",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "focal_distance",
            "value": "focal_distance",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "emission_angle",
            "value": "emission_angle",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "incidence_angle",
            "value": "incidence_angle",
            "type": "parent_property",
            "value_type": "quant",
        },
        {
            "label": "phase_angle",
            "value": "phase_angle",
            "type": "parent_property",
            "value_type": "quant",
        },
    ]

    graphable_properties = [
        prop
        for prop in accessible_properties
        if prop["value"] not in ("color", "seq_id", "name")
    ]
    searchable_fields = [
        prop
        for prop in accessible_properties
        if (prop["type"] != "method") and prop["value"] not in "ltst"
    ]

    def filter_values(
        self,
        scale_to: Optional[Sequence] = None,
        average_filters: bool = False,
    ) -> dict[str, dict]:
        """
        return dictionary of filter values, optionally scaled and merged
        according to MERSPECT-style rules
        scale_to: None or tuple of (lefteye filter name, righteye filter name)
        """
        spectrum = {
            filt: getattr(self, filt.lower()) for filt in self.filters.keys()
        }
        spectrum |= {
            filt + "_ERR": getattr(self, filt.lower() + "_err")
            for filt in self.filters.keys()
        }
        return polish_xcam_spectrum(
            spectrum=spectrum,
            cam_info=DERIVED_CAM_DICT["ZCAM"],
            scale_to=scale_to,
            average_filters=average_filters,
        )

    def image_files(self) -> dict:
        filedict = {
            image_type: getattr(self.observation, image_type)
            for image_type in ZSPEC_IMAGE_TYPES
            if getattr(self.observation, image_type)
        }
        return filedict

    def overlay_browse_file_info(self, image_directory: str) -> dict:
        """
        directory containing browse image files ->
        {'right_file':string, 'left_file': 'left':PIL.Image}
        hack version that just shows rgb and enhanced images
        """
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            browse_filename = (
                fs.path.splitext(filename)[0] + "_browse.jpg"
            )
            images[image_type + "_file"] = browse_filename
            with PIL.Image.open(
                fs.path.join(image_directory, browse_filename)
            ) as image:
                images[image_type + "_size"] = image.size
        return images

    # color corresponding to ROIs drawn on false-color images by MASTCAM team.
    def roi_hex_code(self) -> str:
        return MERSPECT_M20_COLOR_MAPPINGS[self.color]

    def metadata_dict(self) -> dict:
        """
        metadata-summarizing function. could be made more efficient.
        """
        spec_dict = modeldict(self)
        # noinspection PyTypeChecker
        obs_dict = modeldict(self.observation)
        return keyfilter(
            lambda x: x
            in [field["value"] for field in self.searchable_fields],
            spec_dict | obs_dict,
        )