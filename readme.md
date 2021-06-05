M/ZCAM multispectral plotter tool. Beta status.

### dependencies
See the ```conda``` [environment.yml](environment.yml) file. ```conda env create -f environment.yml``` should 
produce an env named ```multidex``` suitable for running the app.

### usage
runtime commands enabling user-facing functionality are currently all included in [multidex.py](multidex/plotter/cruft_holder/multidex.py) 
(MCAM data) and [multidex_z.py](multidex/plotter/cruft_holder/multidex_z.py) (ZCAM data). Run one of these scripts, and a server should 
launch at 127.0.0.1:8051 (MCAM) or :8050 (ZCAM). Should work on any platform ```conda``` works on. Chrome-family 
browsers will probably be faster than Firefox or Safari. We do not recommend accessing an individual server in multiple 
tabs or windows. Behavior in this case is undefined. However, if you launch multiple servers on separate ports -- such 
as the MCAM and ZCAM applications in their default configurations -- accessing them simultaneously should be safe.

### observational data and metadata
Monolithic .sqlite3 database files are distributed separately from this application. They go in 
[multidex/data](multidex/data/) Please contact the repository maintainers if you need and do not have access to these 
files.

### known issues
* PCA is not compatible with 'average nearby filters' mode. It will not work and may break the application.
* the Dockerfile is unfinished. Do not use it.

### licensing notes
This code carries a BSD 3-Clause license. You can do nearly anything that you want with it. _However_, **some of the contents of this repository may be subject to the Mars 2020 Team Guidelines, so please act accordingly.**
