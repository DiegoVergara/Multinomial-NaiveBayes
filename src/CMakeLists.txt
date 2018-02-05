cmake_minimum_required(VERSION 2.8)
project( naive_bayes )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11 -fopenmp")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g -traceback")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
include_directories( "/usr/include/eigen3/" )
find_path(FFTW_INCLUDE_DIR fftw3.h  ${FFTW_INCLUDE_DIRS})
find_library(FFTW_LIBRARY fftw3 ${FFTW_LIBRARY_DIRS})
find_package(OpenCV REQUIRED)

add_executable( test_multinomialNB test_multinomial.cpp multinomialnaivebayes.cpp  multinomial.cpp c_utils.cpp) 
target_link_libraries( test_multinomialNB)

include_directories( "LBP/" )
FILE(GLOB_RECURSE lbp_sourcefiles "LBP/*.cpp")

add_executable( test_adaboost test_adaboost.cpp adaboost.cpp gaussiannaivebayes.cpp multinomialnaivebayes.cpp multinomial c_utils.cpp) 
target_link_libraries( test_adaboost)

add_executable( test_gaussianNB test_gaussian.cpp gaussiannaivebayes.cpp c_utils.cpp) 
target_link_libraries( test_gaussianNB)

add_executable(test_lbp_dataset test_dataset.cpp lbp_dataset.cpp LBP/LBP.cpp) 
target_link_libraries(test_lbp_dataset ${OpenCV_LIBS} ${FFTW_LIBRARY})

add_executable( gender_classification test_clasification.cpp lbp_dataset.cpp LBP/LBP.cpp adaboost.cpp gaussiannaivebayes.cpp multinomialnaivebayes.cpp multinomial c_utils.cpp) 
target_link_libraries( gender_classification ${OpenCV_LIBS} ${FFTW_LIBRARY})