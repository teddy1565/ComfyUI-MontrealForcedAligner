conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_mfa

sudo apt install git g++ make automake autoconf sox libtool subversion \
                 python2.7 python3 zlib1g-dev gfortran wget unzip \
                 libatlas-base-dev
sudo apt install libfst-tools libfst-dev