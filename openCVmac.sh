# Installing OpenCV python libs on mac to work with virtualenv
# OpenCV  2.4.3
# Python 2.7.3 installed with brew

# assuming you have virtualenv, pip, and python installed via brew
# assuming $WORKON_HOME is set to something like ~/.virtualenvs

# using homebrew - make sure we're current
brew update

# setup virtual env
mkvirtualenv opencv
workon opencv

# install numpy
pip install numpy 
pip install scipy
pip install matplotlib
pip install ipython
pip install sphinx
pip install cython
pip install pygments

# requirements for opencv
brew install lame jpeg libpng cmake pkg-config eigen libtiff jasper ffmpeg
brew link jpeg libpng

# Install Xcode 4.6.3 in order to get llvm-gcc working. Not provided with Xcode 5+.

# XX SHA1 mismatch on tbb
mate /usr/local/Library/Formula/tbb.rb

#Update Lines 5 and 6 to latest source and sha1    
  url 'http://threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb41_20121112oss_src.tgz'
  sha1 '752943b78d7a6d3a764feb1bbd7df6c230170cf1'

brew install tbb

# download OpenCV
# wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.3/OpenCV-2.4.3.tar.bz2?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fopencvlibrary%2Ffiles%2Fopencv-unix%2F2.4.3%2F&ts=1353964941&use_mirror=iweb

# unzip
# tar xvfJ OpenCV-2.4.3.tar.bz2

# cd OpenCV-2.4.3/

# Download and cd into the latest OpenCV version
mkdir release
cd release

cmake -D CMAKE_BUILD_TYPE=DEBUG \
-D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python \
-D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D PYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.7.8/Frameworks/Python.framework/Headers \
-D PYTHON_LIBRARY=/usr/local/Cellar/python/2.7.8/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib \
..

make -j8
make install

# cmake -D CMAKE_BUILD_TYPE=DEBUG \
# -D WITH_TBB=OFF \
# -D WITH_OPENGL=ON \
# -D WITH_OPENCL=OFF \
# -D ENABLE_PROFILING=ON \
# -D ENABLE_PRECOMPILED_HEADERS=ON \
# -D ENABLE_COVERAGE=ON \
# -D ENABLE_FAST_MATH=ON \
# -D PYTHON2_EXECUTABLE=$VIRTUAL_ENV/bin/python \
# -D PYTHON2_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages \
# -D INSTALL_PYTHON_EXAMPLES=ON \
# -D PYTHON2_INCLUDE_DIR=/usr/local/Cellar/python/2.7.8/Frameworks/Python.framework/Headers \
# -D PYTHON2_LIBRARY=/usr/local/Cellar/python/2.7.8/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib \
# -D PYTHON3_EXECUTABLE=$VIRTUAL_ENV/bin/python3 \
# -D PYTHON3_PACKAGES_PATH=$VIRTUAL_ENV/lib/python3.4/site-packages \
# -D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/3.4.2_1/Frameworks/Python.framework/Headers \
# -D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.4.2_1/Frameworks/Python.framework/Versions/3.4/lib/libpython3.4.dylib \
# ..