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
We **strongly** advise against installing multiple versions of `conda`
unless you a very specific reason to do so.*

We recommend using 
[Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).
Follow the instructions on that website to download the installer script and 
set up your `conda` installation.

If you have trouble installing `conda`, check "common gotchas" below. If they don't help, 
there are a multitude of helpful tutorials online. [Here is one.](https://www.youtube.com/watch?v=zL65J9c5_KU))

## step 2: create conda environment and install multidex

Now that you have `conda` installed, you can set up a Python environment
to use MultiDEx. Open a terminal window: Anaconda Prompt on Windows, 
Terminal on macOS, or your terminal emulator of choice on Linux. (Windows 
might name the prompt "Miniconda Prompt" or something else instead; just 
search for "prompt" in the Start Menu and don't pick Windows Command Prompt.)

Now, navigate to the directory where you downloaded the repository and run 
the command:

`conda env create -f environment.yml`

If you're on Windows, instead run `conda env create -f windows_environment.yml`

Say yes at the prompts and let the installation finish. Then run:

`conda env list`

You should see `multidex` in the list of environments.

## step 3: activate conda environment and install `multidex`

Next, run:

`conda activate multidex`

and you will be in a Python environment that contains all the packages
MultiDEx needs. 

To install the MultiDEx application into the environment, run 
`pip install -e .` You will never need to run this again unless you delete and
recreate the `multidex` environment.

**Important:** now that you've created this environment, you should always 
have it active whenever you work with MultiDEx. You can do this simply by 
running `conda activate multidex`.

## step 4: put data files in the data directory

The .sqlite3 database files that actually contain the data are distributed 
separately from this application. If you are affiliated with the Mars Science 
Laboratory or Mars 2020 missions and require access to files that contain 
confidential mission data, please contact the repository maintainers at
m20@millionconcepts.com. 

Once you have retrieved whatever .sqlite3 files you will be using, place them 
in the multidex/data subdirectory, which should already contain one file, 
"backend.sqlite3".

## step 5: put browse images in the browse directory

Browse images are also distributed separately from this application as 
.zip files. Please email m20@millionconcepts.com if you need and do not have 
access to a specific set of browse files. Once you have the .zip file for the 
instrument you are working with, uncompress it into 
multidex/plotter/application/assets/browse/$INSTRUMENT_NAME, e.g., 
multidex/plotter/application/assets/browse/mcam.

## step 6: run multidex

Now you can execute MultiDEx by running `multidex INSTRUMENT_CODE`, 
e.g., `multidex MCAM`. (You do not have to be in any specific working directory 
to run this command.) A server should launch at 127.0.0.1:49303. Open that URL
in a browser and you should see the application. You can quit the application 
at any time by pressing Ctrl+C in the terminal window (or simply closing the 
terminal).

**Tip:**
We recommend using Google Chrome or another Chromium-based browser (such as 
Chromium or Microsoft Edge). Although MultiDEx is compatible with Firefox and 
Safari, its performance is better in Chromium-based browsers.

**Tip:**
You can run multiple instances of MultiDEx at once. The easiest way to do this 
is to run `multidex INSTRUMENT_CODE` in multiple terminal windows. Additional
executions of MultiDEx will launch on incrementing port numbers: a second 
execution will launch at 127.0.0.1:49304, a third at 127.0.0.1:49305, and so on.

## common gotchas

* If you get error messages when running MultiDEx, make sure you have activated the `conda` environment by running 
  `conda activate multidex`.
* If terminal commands don't seem like they're working at all, make sure you're in 
  the correct directory -- depending on the task, that might be the directory where you
  downloaded the `conda` setup script, or the directory where you cloned the MultiDEx repository.
  If you're on Windows, make sure you're not using the system Command Prompt or Powershell,
  but instead the Anaconda Prompt (sometimes called Miniconda or Miniforge Prompt, etc.) 
  that came with your`conda` installation.
* If you use multiple shells on MacOS or Linux, `conda` will only 
  automatically set up the one it detects as your system default. If you can't 
  activate the environment, check a different shell or explicitly run `conda init NAME_OF_YOUR_SHELL`
  (`zsh` is the default shell on MacOS).
* If you've already got an installed version of `conda` on your system, installing 
  an additional version without uninstalling the old one may make environment setup very 
  challenging. We do not recommend installing multiple versions of `conda` at once 
  unless you really need to.
