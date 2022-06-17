C/M/ZCAM multispectral plotter tool. Beta status.

### installation
See [the installation guide](installation_guide.md).

### usage
runtime commands enabling user-facing functionality are currently all included in 
[multidex.py](multidex/multidex.py). Run this script with a single argument for the instrument whose data 
you'd like to explore (e.g. `python multidex.py ZCAM`) and a server should launch at 127.0.0.1:49303. Pointing your 
browser at this server will open the application in your browser. This should work on any platform `conda` works on. 
Chrome-family browsers will probably be faster than Firefox or Safari. We do not recommend accessing an individual 
server in multiple tabs or windows. Behavior in this case is undefined. However, if you run the script multiple times
from the command line, it will open additional servers, incrementing the port number by one each time. Accessing these
servers separately should be completely safe.

More detailed usage instructions, including a user manual and tutorials, are 
forthcoming.

### observational data and metadata
Monolithic .sqlite3 database files and compressed browse images are 
distributed separately from this application. There is one publicly-available 
data set available: MSL Mastcam up to sol 2300. 
[You can find it here.](https://drive.google.com/drive/folders/1478lDoe1fOmQAWO_8Nl77-GX46Iz9Np1)
If you are affiliated with the Mars Science Laboratory or Mars 2020 missions and
require access to MultiDEx files that contain confidential data,
please contact the repository maintainers.

### known issues
* the spectra-info dialog is not currently properly draggable on Firefox. A fix is planned.
* image export does not work on Windows. A fix for this is also planned.

### licensing notes
This code carries a BSD 3-Clause license. You can do nearly anything that 
you want with it. _However_, **some data referenced by code in this 
repository may be subject to the MSL and/or Mars 2020 Team Guidelines, so 
please act accordingly.**
