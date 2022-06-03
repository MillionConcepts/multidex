# MultiDEx installation guide

*Note: if anything doesn't work, see 'common gotchas' at the bottom of this document.*

## step 0: clone the repository to your computer

If you have `git` installed on your computer, navigate in a terminal emulator to wherever you'd 
like to place the software and run `git clone https://github.com/MillionConcepts/multidex.git`.
If you don't, and you are on Windows or MacOS, we recommend using 
[GitHub Desktop](https://desktop.github.com/). Install that program, run it, 
log in to your account, choose "clone a repository from the Internet," click the "URL" tab,
paste `https://github.com/MillionConcepts/multidex.git` into the 'Repository URL' field,
and click 'Clone'.

## step 1: install conda

*Note: If you already have Anaconda or Miniconda installed on your computer, you can
skip this step. If it's very old or not working well, you should uninstall it first.
We **definitely** don't recommend installing multiple versions of `conda`
unless you have a strong and specific reason to do so.*

We recommend using [Mambaforge](https://github.com/conda-forge/miniforge). 
Download the appropriate version of the installer for your operating system and 
processor architecture (in most cases 64-bit). If you are on Windows, just double-click
the .exe to start installation; on MacOS or Linux, navigate to wherever you downloaded
the file in a terminal and run "sudo chmod +x name_of_file.sh" followed by 
"./name_of_file.sh". If this doesn't work, try running the commands in that website.

It you don't want to use Mambaforge, 
[you can get `conda` here as part of the Miniconda distribution of Python](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html).
Download the appropriate version of the installer and follow the instructions on that 
website to set up your `conda` installation. Make sure you download Miniconda3, not 
Miniconda2. `multidex` is not compatible with Python 2.

**IMPORTANT: If you install Miniconda, replace `mamba` in all the commands below with `conda`.**

If you have trouble installing `conda`, check "common gotchas" below. If they don't help, 
there are a multitude of helpful tutorials online. [Here is one.](https://www.youtube.com/watch?v=zL65J9c5_KU))

## step 2: create conda environment

Now that you have `conda` installed, you can set up a Python environment
to use MultiDEx. Open a terminal window: Anaconda Prompt on Windows, Terminal on macOS,
or your terminal emulator of choice on Linux. 

On MacOS or Linux, navigate to the directory where you put the repository and run the command:
`mamba env create -f environment.yml`

On Windows, you will need to install ImageMagick separately. 
[You can get it here.](https://download.imagemagick.org/ImageMagick/download/binaries/ImageMagick-7.0.10-62-Q16-HDRI-x64-dll.exe) 
When installing ImageMagick, **you must click the "install development headers and libraries 
for C and C++" box in the installer.** If you do not do this, the Wand library will be unable 
to find your installation of ImageMagick. After installing ImageMagick, run 
`mamba env create -f windows_environment.yml`.

## step 3: activate conda environment

Say yes at the prompts and let the installation finish. Then run:

`mamba env list`

You should see `multidex` in the list of environments. Now run:

`conda activate multidex`

and you will be in a Python environment that contains all the packages
MultiDEx needs. 

**Important:** now that you've created this environment, you should 
always have it active whenever you work with MultiDEx.

## step 4: put data files in the data directory

The monolithic .sqlite3 database files that actually contain the data are 
distributed separately from this application. Please contact the repository 
maintainers if you need and do not have access to these files. Once you have
these files, place them in the multidex/data subdirectory, which should already
contain one file, "backend.sqlite3".

## step 5: put browse images in the browse directory

Browse images are also distributed separately from this application as .zip files. Please contact 
the repository maintainers if you need and do not have access to these files. Once you
have the .zip file for the instrument you are working with, uncompress it into 
multidex/plotter/application/assets/browse/$INSTRUMENT_NAME, e.g., 
multidex/plotter/application/assets/browse/mcam.

## step 6: run multidex

Now you can execute MultiDEx by running `python multidex.py INSTRUMENT_CODE`, 
e.g., `python multidex.py MCAM`. A server should launch at 127.0.0.1:49303.
Open that URL in a browser and you should see the application. We recommend
Chrome-family browsers, although it should function in Firefox or Safari --
it will just be slower.

## common gotchas

* If you get error messages when running MultiDEx, 
  make sure you have activated the `conda` environment by running `conda activate multidex`.
* If terminal commands don't seem like they're working at all, make sure you're in 
  the correct directory -- depending on the task, that might be the directory where you
  downloaded the `conda` setup script, or the directory where you cloned the MultiDEx repository.
  If you're on Windows, make sure you're not using the system Command Prompt or Powershell,
  but rather the Anaconda Prompt (sometimes called Miniconda Prompt, etc.) that came with your
  `conda` installation.
* If you use multiple shells on macOS or Linux, `conda` will only 
  automatically set up the one it detects as your system default. If you can't 
  activate the environment, check a different shell or explicitly run `conda init NAME_OF_YOUR_SHELL`
* If you've already got an installed version of `conda` on your system, installing 
  an additional version without uninstalling the old one may make environment setup very 
  challenging. We do not recommend installing multiple versions of `conda` at once 
  unless you really need to.
