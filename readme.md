M/ZCAM multispectral plotter tool. Late alpha, not for distribution.

### dependencies
See the ```conda``` [environment.yml](environment.yml) file. ```conda env create -f environment.yml``` should 
produce an env named ```multidex``` suitable for running the app.

### usage
runtime commands enabling user-facing functionality are currently all included in [multidex.py](multidex/multidex.py). Run that script, and a server should launch at 127.0.0.1:8050. 
Should work on any platform ```conda``` works on. Chrome-family browsers will probably be faster than Firefox or Safari.
