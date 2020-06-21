from django.db import models
import pandas as pd

# Create your models here.

class Observation(models.Model):
    """
    class representing a single observation consisting of one or more images of the same scene,
    possibly created from a multispectral series of images, 
    possibly along with named ROIs that correspond to reduced spectral data contained in Spectrum
    objects. ROIs as drawn on each image notionally correspond to the same physical locations. 

    this is a parent class intended to be subclassed for individual instruments.
    """

    name = models.CharField("Name", max_length=100, db_index=True)

    def spectra(self):
        return self.spectra_set.all()
        

class Spectrum(models.Model):
    """
    class representing reduced spectral data derived from a named ROI on one or more images.

    this is a parent class intended to be subclassed for individual instruments.
    """

    observation_class = Observation
    observation = models.ForeignKey(observation_class, on_delete=models.PROTECT, related_name="spectra_set", blank=True)
    
    # this is not intended to hold all responsivity information about the filters,
    # unlike FilterSet from wwu_spec (or some version that may be instantiated here later)
    # it is primarily a container.
    # so just: filter names and canonical center values.

    filters = {}

    # this will sometimes be faster and sometimes slower than just directly accessing database fields
    def as_table(self):
        rows = []
        for filt, freq in self.filters.items():
            # _maybe_ try-except -- but perhaps we don't want to be tolerant of missing info
            mean = self.getattr(filt + "_mean")
            stdev = self.getattr(filt + "_stdev")
            rows.append([filt, freq, mean, stdev])
        return pd.DataFrame(rows) 


class MObs(Observation):
    """Observation subclass for MASTCAM"""

    sol = models.IntegerField("Sol", db_index = True)
    # note this is ltst for first frame of sequence (usually left eye 'clear')
    ltst = models.TimeField("Local True Solar Time", db_index = True)
    # not sure what this actually is. format is of sequence
    # number in PDS header, but _value_ corresponds to request id in the PDS header
    mcam = models.CharField("mcam", max_length=20, db_index=True)

    rover_el = models.FloatField("Rover Elevation", blank=True, db_index=True)
    target_el = models.FloatField("Target Elevation", null=True, db_index=True)
    tau = models.FloatField("Interpolated Tau", blank=True, db_index=True)
    # this field is duplicated in Tina's sample data. assuming for now
    # that this is a mistake.
    f_dist = models.FloatField("Focal Distance", blank=True, db_index=True)
    i_angle = models.FloatField("Incidence Angle", blank=True, db_index=True)
    e_angle = models.FloatField("Emission Angle", blank=True, db_index=True)
    so_lon = models.FloatField("Solar Longitude", blank=True, db_index=True)
    # i think an arbitrary number for the site. part of the ROVER_NAV_FRAME
    # coordinate system
    site = models.IntegerField("Site", blank=True, db_index=True)
    # similar
    drive = models.IntegerField("Drive", blank=True, db_index=True)

    # presumably planetodetic lat/lon
    # not in the image labels in PDS

    lat = models.FloatField("Latitude", blank=True, db_index=True)
    lon = models.FloatField("Longitude", blank=True, db_index=True)

    # don't know what this is
    traverse = models.FloatField("Traverse", blank=True, db_index=True)

    # the right-eye images have a narrower FOV than the left-eye images, and two RGB
    # right-eye images are produced for each observation.
    # for some but not all observations, two corresponding spectrum-reduced right-eye images are also
    # produced. In this case, a duplicate spectrum-reduced left-eye image is also produced for ROI matching
    # (presumably due to limitations in MERTOOLS). We duplicate that structure here.

    righteye_s_image_1 = models.CharField("path to first spectrum-reduced right-eye image", blank=True, max_length=100, db_index=True)
    lefteye_s_image_1 = models.CharField("path to first spectrum-reduced left-eye image", blank=True, max_length=100, db_index=True)
    righteye_s_image_2 = models.CharField("path to second spectrum-reduced right-eye image", blank=True, max_length=100, db_index=True)
    lefteye_s_image_2 = models.CharField("path to second spectrum-reduced left-eye image", blank=True, max_length=100, db_index=True)
    righteye_rgb_image_1 = models.CharField("path to first rgb right-eye image", blank=True, max_length=100, db_index=True)
    lefteye_rgb_image_1 = models.CharField("path to rgb left-eye image", blank=True, max_length=100, db_index=True)
    righteye_rgb_image_2 = models.CharField("path to second rgb right-eye image", blank=True, max_length=100, db_index=True)

    def __str__(self):
        return "sol"+self.sol+"_"+self.mcam


class MSpec(Spectrum):
    """Spectrum subclass for MASTCAM"""

    # feature type (rock, dust, etc) (not apparently exported with Tina's sample data)
    feature_type = models.CharField("Feature Type", blank=True, max_length=50, db_index = True)

    ###########################################################
    ### relationships with parent observation and images
    ###########################################################

    observation_class = MObs

    # color of associated ROI
    roi_color = models.CharField("ROI Color", blank=True, max_length=20, db_index=True)

    # for some but not all observations, two spectrum-reduced right-eye images are produced. 
    # In this case, a duplicate spectrum-reduced left-eye image is also produced for ROI matching.
    # this field indicates which image the ROI is drawn on
    image_number = models.IntegerField("image number", default=1, db_index=True)

    #############################################
    ### lithological information -- relevant only to rocks ###
    ##########################################################

    is_floating = models.BooleanField("floating vs. in-place", null=True, db_index=True)
    
    # large-to-small taxonomic categories for rock clusters
    group = models.CharField("Group", blank=True, max_length=50, db_index = True)
    formation = models.CharField("Formation", blank=True, max_length=50, db_index = True)
    member = models.CharField("Member", blank=True, max_length=50, db_index = True)

    ### end lithological ###

    notes = models.CharField("Notes", blank=True, max_length=100, db_index=True)

    ##############################################
    ########### explicit fields for each filter
    ##############################################

    # i'm pretty sure multiple fields is better than a stringified dict
    # it permits SQL queries on spectra
    # and doesn't require thousands of fields b/c not high-res spectra

    # this is _extremely_ wordy. there may be an equally-safe
    # way to do this by iterating over filter dictionary,
    # but I am nervous about it because every change can create SQL migrations,
    # and the question of when properties are set is dicey.

    L0_blue_mean = models.FloatField("L0 (Blue Bayer) mean", db_index = True)
    L0_blue_stdev = models.FloatField("L0 (Blue Bayer) stdev", db_index = True)
    L0_green_mean = models.FloatField("L0 (Green Bayer) mean", db_index = True)
    L0_green_stdev = models.FloatField("L0 (Green Bayer) stdev", db_index = True)
    L0_red_mean = models.FloatField("L0 (Red Bayer) mean", db_index = True)
    L0_red_stdev = models.FloatField("L0 (Red Bayer) stdev", db_index = True)
    R0_blue_mean = models.FloatField("R0 (Blue Bayer) mean", db_index = True)
    R0_blue_stdev = models.FloatField("R0 (Blue Bayer) stdev", db_index = True)
    R0_green_mean = models.FloatField("R0 (Green Bayer) mean", db_index = True)
    R0_green_stdev = models.FloatField("R0 (Green Bayer) stdev", db_index = True)
    R0_red_mean = models.FloatField("R0 (Red Bayer) mean", db_index = True)
    R0_red_stdev = models.FloatField("R0 (Red Bayer) stdev", db_index = True)
    R1_mean = models.FloatField("R1 mean", db_index = True)
    R1_stdev = models.FloatField("R1 stdev", db_index = True)
    L1_mean = models.FloatField("L1 mean", db_index = True)
    L1_stdev = models.FloatField("L1 stdev", db_index = True)
    R2_mean = models.FloatField("R2 mean", db_index = True)
    R2_stdev = models.FloatField("R2 stdev", db_index = True)
    L2_mean = models.FloatField("L2 mean", db_index = True)
    L2_stdev = models.FloatField("L2 stdev", db_index = True)
    R3_mean = models.FloatField("R3 mean", db_index = True)
    R3_stdev = models.FloatField("R3 stdev", db_index = True)
    L3_mean = models.FloatField("L3 mean", db_index = True)
    L3_stdev = models.FloatField("L3 stdev", db_index = True)
    R4_mean = models.FloatField("R4 mean", db_index = True)
    R4_stdev = models.FloatField("R4 stdev", db_index = True)
    L4_mean = models.FloatField("L4 mean", db_index = True)
    L4_stdev = models.FloatField("L4 stdev", db_index = True)
    R5_mean = models.FloatField("R5 mean", db_index = True)
    R5_stdev = models.FloatField("R5 stdev", db_index = True)
    L5_mean = models.FloatField("L5 mean", db_index = True)
    L5_stdev = models.FloatField("L5 stdev", db_index = True)
    R6_mean = models.FloatField("R6 mean", db_index = True)
    R6_stdev = models.FloatField("R6 stdev", db_index = True)
    L6_mean = models.FloatField("L6 mean", db_index = True)
    L6_stdev = models.FloatField("L6 stdev", db_index = True)

    filters = {
        'L2':445,
        'R2':447,
        'L0_blue':495,
        'R0_blue':495,
        'L1':527,
        'R1':527,
        'L0_green':554,
        'R0_green':554,
        'L0_red':640,
        'R0_red':640,
        'L4':676,
        'L3':751,
        'R3':805,
        'L5':867,
        'R4':908,
        'R5':937,
        'L6':1012,
        'R6':1013
        }
