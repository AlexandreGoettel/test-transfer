#!/bin/sh
################################################################################
# Script to start the CMake-driven rebuild process for the entire 
# project. The project will be rebuilt in a directory 'build'; any preexisting 
# directory 'build' will be deleted. 
################################################################################

# define parent directory for the simulation build and install
if [ -z $SIM_DIR ]; then
	SIM_DIR=$(pwd)
fi

# get script name and call directory
 ME="$(basename "$(readlink -f "$BASH_SOURCE")")"
 ME="[$ME]"
 FROM=$PWD
 
 echo "$ME Starting to build project..."
 echo "$ME SIM_DIR: $SIM_DIR"

# replace existing 'build' directory in the project's base directory
 echo "$ME Going to directory where to build the project."
 cd $SIM_DIR
 echo "$ME Removing existing 'build' directory."
 rm -rf build
 echo "$ME Removing existing 'install' directory."
 rm -rf install
 echo "$ME Creating new 'build' directory."
 mkdir build
 echo "$ME Creating new 'install' directory."
 mkdir install
 echo "$ME Going into 'build' directory."
 cd build
 
# build with CMAKE; 
 echo "$ME Calling CMake to build."
 cmake -DCMAKE_INSTALL_PREFIX=$SIM_DIR/install \
       -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
       -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release \
       $SIM_DIR
 cmake --build . -j 4

# install with CMAKE
echo "$ME Calling CMake to install."
  cmake --build . --target install
   
# go back to call directory
 echo "$ME Going back to call directory."
 cd $FROM
