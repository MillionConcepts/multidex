working area for MASTCAM spectral plotter tool rewrite. preliminary scratch work. not for distribution.

* sensible SQL tables and associated data structures for MASTCAM observations and spectra.
	* Spectrum objects include API for user-facing spectroscopy functions (band depth, etc)
	* current database contains sample data exported by Tina.
* scripts that ingest MASTCAM spectra archive (at least in the format exported by Tina. hopefully the full archive is similar!)
* a growing library of objects and interaction patterns for graphing these spectra
	* 'axis property' selection objects that call calculations in the Spectrum object API or fetch other properties
	* a simple search interface that allows arbitrary numbers of filters and supports (autopopulated) string and numeric search across many fields
	* simple text display of point properties
* caching system for sharing data between dash app objects

