# [cmake](https://github.com/iLovEing/notebook/issues/34)

## cmake example
```
#[[
## version 1: build with src file
cmake_minimum_required(VERSION 3.20)  # cmake version
set(CMAKE_CXX_STANDARD 17)  # c++ standard

project(CMAKE_TEST)  # project name
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)  # executable output path
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)  # library output path

# lib is different between linux and windows, building is decided by toolchain.
# e.g. mingw64-win32 and mingw64-posix
include_directories(${PROJECT_SOURCE_DIR}/inc)  # add head file dir
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)  # search src file, GLOB_RECURSE for recursive search
add_executable(main main.c ${SRC_LIST})  # build executable
add_library(cmake_test_lib_static STATIC main.c ${SRC_LIST})  # build static lib
add_library(cmake_test_lib_share SHARED main.c ${SRC_LIST})  # build shared lib
]]


#[[
## version 2: build with static lib
cmake_minimum_required(VERSION 3.20)  # cmake version
set(CMAKE_CXX_STANDARD 17)  # c++ standard

project(CMAKE_TEST)  # project name
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)  # executable output path
add_definitions(-DTEST_DEFINE)  # add define

include_directories(${PROJECT_SOURCE_DIR}/inc)  # add head file dir
link_directories(${PROJECT_SOURCE_DIR}/lib)  # add lib path if library is not system lib
link_libraries(cmake_test_lib_static)  # link before build executable
add_executable(main_link_static main.c)  # build executable
]]


#[[
## version 3: build with share lib, it seems not work on windows (cannot find dll)
cmake_minimum_required(VERSION 3.20)  # cmake version
set(CMAKE_CXX_STANDARD 17)  # c++ standard

project(CMAKE_TEST)  # project name
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)  # executable output path

include_directories(${PROJECT_SOURCE_DIR}/inc)  # add head file dir
link_directories(${PROJECT_SOURCE_DIR}/lib)  # add lib path if library is not system lib
add_executable(main_link_shared main.c)  # build executable
target_link_libraries(main_link_shared cmake_test_lib_share)  # link after build executable
]]
```