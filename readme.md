Working area for MASTCAM spectral plotter tool rewrite. Pre-alpha, not for distribution.

### dependencies
Formal dependency list not yet generated. Complete list of packages installed in development environment is in [env_list](env_list).

### usage
runtime commands enabling user-facing functionality are currently all included in [app.py](mastspec/app.py). Run the script, and a server should launch at 127.0.0.1:8050. (Note that app.ipynb is currently deprecated.)

### current features
* sensible SQL tables and associated data structures for MASTCAM observations and spectra
	* Spectrum objects include API for user-facing spectroscopy functions (band depth, etc)
	* current database contains the full MASTCAM archive as of Melissa's December 2020 delivery, placed into official archival format by Chase's [mcam_spect_data_conversion](https://github.com/MillionConcepts/mcam_spect_data_conversion) module. See [ingest_csv.ipynb](mastspec/ingest_csv.ipynb) for modifications / preprocessing of this archival format for Django/SQLite ingestion, along with association of MSpec objects with image files. 
	* a variety of extensible support functions
* a growing library of objects and interaction patterns for graphing these spectra
	* 'axis property' selection objects that call calculations in the Spectrum object API or fetch other properties
	* a simple search interface that allows arbitrary numbers of filters and supports (autopopulated) string and numeric search across many fields
	* individual spectral graphs and simple text display of spectra properties
	* thumbnail image display for individual spectra
	* tabbed graph-detail dialogs
	* search state saving
	* the beginnings of a marker styling system