# General config
cmake_minimum_required(VERSION 3.4.1 FATAL_ERROR)
project(LPSD
	VERSION 0.0.1
	DESCRIPTION "2023 Adaptation of LPSD Software for scalability")

# Add HDF5
SET(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 REQUIRED)
message(STATUS "HDF5 INCLUDE DIR: " ${HDF5_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})

# Add FFTW3
#find_package(PkgConfig REQUIRED)
#pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
#include_directories(PkgConfig::FFTW)

# Add GSL
find_package(GSL REQUIRED)

# Set variables
SET(EXENAME "lpsd-exec")
SET(INCLUDEPATH ${CMAKE_CURRENT_SOURCE_DIR})
SET(SRCPATH ${CMAKE_CURRENT_SOURCE_DIR})
SET(SOURCE
	${SRCPATH}/IO.c
	${SRCPATH}/config.c
	${SRCPATH}/debug.c
	${SRCPATH}/errors.c
	${SRCPATH}/lpsd-exec.c
	${SRCPATH}/lpsd.c
	${SRCPATH}/misc.c
	${SRCPATH}/tics.c
	${SRCPATH}/genwin.c
	${SRCPATH}/ArgParser.c
	${SRCPATH}/StrParser.c
	${SRCPATH}/netlibi0.c
	${SRCPATH}/ask.c
	${SRCPATH}/fft.c
)
SET(HEADERS
	${INCLUDEPATH}/IO.h
	${INCLUDEPATH}/config.h
	${INCLUDEPATH}/debug.h
	${INCLUDEPATH}/errors.h
	${INCLUDEPATH}/lpsd-exec.h
	${INCLUDEPATH}/lpsd.h
	${INCLUDEPATH}/misc.h
	${INCLUDEPATH}/tics.h
	${INCLUDEPATH}/genwin.h
	${INCLUDEPATH}/ArgParser.h
	${INCLUDEPATH}/StrParser.h
	${INCLUDEPATH}/netlibi0.h
	${INCLUDEPATH}/ask.h
	${INCLUDEPATH}/fft.h
)

# Set executable(s)
add_executable(${EXENAME} ${SOURCE} ${HEADERS})

# Link & install
target_link_libraries(${EXENAME} PRIVATE HDF5::HDF5 GSL::gsl GSL::gslcblas)# PkgConfig::FFTW)
target_include_directories(${EXENAME} PRIVATE ${INCLUDEPATH})
INSTALL(TARGETS ${EXENAME} DESTINATION bin)

