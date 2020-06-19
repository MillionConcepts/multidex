from django.db import models
import pandas as pd

# Create your models here.

class Observation(models.Model):
	"""
	class representing a single observation consisting of one or more images of the same scene,
	possibly along with named ROIs that correspond to reduced spectral data contained in Spectrum
	objects. ROIs as drawn on each image notionally correspond to the same physical locations. 

	this is a parent class intended to be subclassed for individual instruments.
	"""

	name = models.CharField("Name", max_length=100, db_index = True)


	# ForeignKey fields representing ROI names and corresponding spectra
	roi_fields = []

	def spectra(self):
		return [self.roi for roi in self.roi_fields if self.roi]

		

class Spectrum(models.Model):
	"""
	class representing reduced spectral data derived from a named ROI on one or more images.

	this is a parent class intended to be subclassed for individual instruments.
	"""
	
	# this is not intended to hold all responsivity information about the filters,
	# unlike FilterSet from wwu_spec (or some version that may be instantiated here later)
	# it is primarily a container.
	# so just: filter names and canonical center values.

 	filters = {}

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

	sol = models.IntField("Sol", db_index = True)
	# note this is ltst for _start_ of observation
	ltst = models.TimeField("Local True Solar Time", db_index = True)
	# not sure what this actually is. format is of sequence
	# number in PDS header, but _value_ corresponds to request id in the PDS header
	mcam = models.CharField("mcam", max_length=20, db_index=True)

	rover_el = models.FloatField("Rover Elevation", blank=True, db_index=True)
	target_el = models.FloatField("Target Elevation", blank=True, db_index=True)
	tau = models.FloatField("Interpolated Tau", blank=True, db_index=True)
	# this field is duplicated in Tina's sample data. assuming for now
	# that this is a mistake.
	f_dist = models.FloatField("Focal Distance", blank=True, db_index=True)
	i_angle = models.FloatField("Incidence Angle", blank=True, db_index=True)
	e_angle = models.FloatField("Emission Angle", blank=True, db_index=True)
	so_lon = models.FloatField("Solar Longitude", blank=True, db_index=True)
	# i think an arbitrary number for the site. part of the ROVER_NAV_FRAME
	# coordinate system
	site = models.IntField("Site", blank=True, db_index=True)
	# similar
	drive = models.IntField("Drive", blank=True, db_index=True)

	# presumably planetodetic lat/lon
	# not in the image labels in PDS

	lat = models.FloatField("Latitude", blank=True, db_index=True)
	lon = models.FloatField("Longitude", blank=True, db_index=True)

	# don't know what this is
	traverse = models.FloatField("Traverse", blank=True, db_index=True)

	# MASTCAM ROIs are named according to the color of the polygon that represented the ROI 
	# in MERtools. 
	red = models.ForeignKey("red ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	light_blue = models.ForeignKey("light blue ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	blue = models.ForeignKey("blue ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	teal = models.ForeignKey("teal ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	dark_blue = models.ForeignKey("dark blue ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	dark_red = models.ForeignKey("dark red ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	green = models.ForeignKey("green ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	dark_green = models.ForeignKey("dark green ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	light_green = models.ForeignKey("light green ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	goldenrod = models.ForeignKey("goldenrod ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	yellow = models.ForeignKey("yellow ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	sienna = models.ForeignKey("sienna ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)
	pink = models.ForeignKey("pink ROI", on_delete=models.PROTECT, related_name="observation", blank = True, db_index=True)

	roi_fields = [
		red, light_blue, blue, teal, dark_blue, dark_red, green, dark_green,
		light_green, goldenrod, yellow, sienna, pink
	]

class MSpec(Spectrum):
	"""Spectrum subclass for MASTCAM"""

	# feature type (rock, dust, etc) (not apparently exported with Tina's sample data)
	feature_type = models.CharField("Feature Type", blank=True, max_length=50, db_index = True)


	# color of associated ROI -- redundant with reverse relation
	# but faster in some cases

	roi_color = models.CharField("ROI Color", blank=True, max_length=20, db_index=True)

	### lithological information -- relevant only to rocks ###

	float_status = models.BoolField("floating vs. in-place", blank=True, db_index=True)
	
	# large-to-small taxonomic categories for rock clusters
	group = models.CharField("Group", blank=True, max_length=50, db_index = True)
	formation = models.CharField("Formation", blank=True, max_length=50, db_index = True)
	member = models.CharField("Member", blank=True, max_length=50, db_index = True)

	### end lithological ###

	notes = models.CharField("Notes", blank=True, max_length=100, db_index = True)

	# explicit fields for each filter
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
	R6_mean = models.FloatField("R5 mean", db_index = True)
	R6_stdev = models.FloatField("R5 stdev", db_index = True)
	L6_mean = models.FloatField("L5 mean", db_index = True)
	L6_stdev = models.FloatField("L5 stdev", db_index = True)

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
