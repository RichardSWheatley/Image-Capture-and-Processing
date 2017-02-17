name: py27_32
channels:
- menpo
- defaults
dependencies:
- cycler=0.10.0=py27_0
- enum34=1.1.6=py27_0
- icu=57.1=vc9_0
- jpeg=8d=vc9_2
- libpng=1.6.27=vc9_0
- mkl=2017.0.1=0
- numpy=1.11.3=py27_0
- openssl=1.0.2j=vc9_0
- pip=9.0.1=py27_1
- pyparsing=2.1.4=py27_0
- pyqt=5.6.0=py27_1
- python=2.7.5=2
- python-dateutil=2.6.0=py27_0
- pytz=2016.10=py27_0
- qt=5.6.2=vc9_0
- setuptools=27.2.0=py27_1
- sip=4.18=py27_0
- six=1.10.0=py27_0
- tk=8.5.18=vc9_0
- wheel=0.29.0=py27_0
- zlib=1.2.8=vc9_3
- opencv=2.4.11=py27_1
- opencv3=3.1.0=py27_0
- pip:
  - opencv-python==3.1.0.3

To recreate the anaconda environment.
Install Anaconda for Windows 32-bit.

From the Windows command line (not the python shell):

set CONDA_FORCE_32bit=1

activate py27_32


(py27_32) > python vid_capture_ORB_detector.py