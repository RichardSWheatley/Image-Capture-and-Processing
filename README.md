# Image-Capture-and-Processing
32-bit python2.7.5
opencv3.1.0
numpy1.11.3
enum1.1.6

To recreate the anaconda environment.
Install Anaconda for Windows..any version. I installed windows 64-bit.

From the Windows command line (not the python shell):

conda create -n opencv_env numpy scipy scikit-learn matplotlib python=2

activate opencv_env


From the new opencv_env command propmt:

conda install -c menpo opencv3=3.1.0


then:

python opencv2_vid_capture_gray.py

