cmake_minimum_required (VERSION 2.6)
project (EXACT)

# The version number.
set (EXACT_VERSION_MAJOR 0)
set (EXACT_VERSION_MINOR 33)

#add_definitions( -DEXACT_VERSION="${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR}" )

SET (PLATFORM 64)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -fsanitize=address -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -DNAN_CHECKS")
SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funroll-loops -DNDEBUG")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

message(STATUS "project source dir is ${PROJECT_SOURCE_DIR}")

include_directories(${PROJECT_SOURCE_DIR})

FIND_PATH(TAO_INCLUDE_DIR ./examples/benchmarks.hxx
    ../tao #add addiitonal paths here if undvc_common is in a different directory
    )

IF (${TAO_INCLUDE_DIR} STREQUAL "TAO_INCLUDE_DIR-NOTFOUND") 
    SET (TAO_FOUND FALSE)
    MESSAGE(STATUS "TAO NOT FOUND!")
    SET(TAO_INCLUDE_DIR "")

ELSE (${TAO_INCLUDE_DIR} STREQUAL "TAO_INCLUDE_DIR-NOTFOUND") 
    SET (TAO_FOUND TRUE)

    MESSAGE(STATUS "TAO include directory: ${TAO_INCLUDE_DIR}")

    FIND_LIBRARY(TAO_MPI_LIBRARY NAMES mpi_algorithms PATHS ../tao/build/mpi PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_SYNCH_LIBRARY NAMES synchronous_algorithms PATHS ../tao/build/synchronous_algorithms PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_ASYNCH_LIBRARY NAMES asynchronous_algorithms PATHS ../tao/build/asynchronous_algorithms PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_DB_ASYNCH_LIBRARY NAMES db_asynchronous_algorithms PATHS ../tao/build/asynchronous_algorithms PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_NN_LIBRARY NAMES neural_networks PATHS ../tao/build/neural_networks PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_UTIL_LIBRARY NAMES tao_util PATHS ../tao/build/util  PATH_SUFFIXES lib)
    set(TAO_LIBRARY ${TAO_MPI_LIBRARY} ${TAO_SYNCH_LIBRARY} ${TAO_ASYNCH_LIBRARY} ${TAO_DB_ASYNCH_LIBRARY} ${TAO_NN_LIBRARY} ${TAO_UTIL_LIBRARY})
    MESSAGE(STATUS "TAO library: ${TAO_LIBRARY}")

    include_directories(${TAO_INCLUDE_DIR})
ENDIF (${TAO_INCLUDE_DIR} STREQUAL "TAO_INCLUDE_DIR-NOTFOUND") 

find_package(MPI)

IF (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

    include_directories(${MPI_INCLUDE_PATH})

    add_executable(reca reca)
    target_link_libraries(reca ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TAO_LIBRARY} pthread)
ENDIF(MPI_FOUND)
