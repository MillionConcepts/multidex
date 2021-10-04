M/ZCAM multispectral plotter tool. Beta status.

### dependencies
See the `conda` [environment.yml](environment.yml) file. 

On MacOS and Linux, `conda env create -f environment.yml` should produce an env named `multidex` suitable for running the app. 

On Windows, you will need to install ImageMagick separately. 
[You can get it here.](https://download.imagemagick.org/ImageMagick/download/binaries/ImageMagick-7.0.10-62-Q16-HDRI-x64-dll.exe) 
When installing ImageMagick, **you must click the "install development headers and libraries for C and C++" box in the installer.**
If you do not do this, the Wand library will be unable to find your installation of ImageMagick. After installing ImageMagick,
run `conda env create -f windows_environment.yml`.

### usage
runtime commands enabling user-facing functionality are currently all included in 
[multidex.py](multidex/multidex.py). Run this script with a single argument for the instrument whose data 
you'd like to explore (e.g. `python multidex.py ZCAM`) and a server should launch at 127.0.0.1:49303. Pointing your 
browser at this server will open the application in your browser. This should work on any platform `conda` works on. 
Chrome-family browsers will probably be faster than Firefox or Safari. We do not recommend accessing an individual 
server in multiple tabs or windows. Behavior in this case is undefined. However, if you run the script multiple times
from the command line, it will open additional servers, incrementing the port number by one each time. Accessing these
servers separately should be completely safe.

### observational data and metadata
Monolithic .sqlite3 database files are distributed separately from this application. They go in 
[multidex/data](multidex/data/) Please contact the repository maintainers if you need and do not have access to these 
files.

### known issues
* the spectra-info dialog is not currently properly draggable on Firefox. A fix is planned.

### licensing notes
This code carries a BSD 3-Clause license. You can do nearly anything that you want with it. _However_, **some of the 
contents of this repository may be subject to the Mars 2020 Team Guidelines, so please act accordingly.**
