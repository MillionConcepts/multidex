Working area for MASTCAM spectral plotter tool rewrite. Pre-alpha, not for distribution.

### dependencies
fixed dependency list not yet generated. complete list of packages installed in production environment is in 'env_list'

### usage
runtime commands enabling user-facing functionality are currently all included in app.ipynb. Run cells 1-11, and a server should run at 127.0.0.1:8050 

### current features
* sensible SQL tables and associated data structures for MASTCAM observations and spectra.
	* Spectrum objects include API for user-facing spectroscopy functions (band depth, etc)
	* current database contains sample data exported by Tina.
	* a variety of extensible support functions
* scripts that ingest MASTCAM spectra archive (at least in the format exported by Tina. hopefully the full archive is similar!)
* a growing library of objects and interaction patterns for graphing these spectra
	* 'axis property' selection objects that call calculations in the Spectrum object API or fetch other properties
	* a simple search interface that allows arbitrary numbers of filters and supports (autopopulated) string and numeric search across many fields
	* individual spectral graphs and simple text display of spectra properties
	* display of thumbnail images for individual spectra
	* modal / tabbed graph-detail dialogs
	* etc.
* caching system for sharing data between dash app objects

