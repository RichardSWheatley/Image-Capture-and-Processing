# Image-Capture-and-Processing
32-bit python2.7.5
opencv3.1.0
numpy1.11.3
enum1.1.6

To recreate the anaconda environment.
Install Anaconda for Windows 32-bit.

From the Windows command line (not the python shell):

set CONDA_FORCE_32bit=1

conda create -n py_275 python=2.7.5

activate py_275


From the new opencv_env command propmt:

conda install numpy=1.11.3

conda install enum34

conda install -c menpo opencv3=3.1.0


then:

python vid_capture.py

