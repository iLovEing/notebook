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


#[[
## version 3: build with share lib, it seems not work on windows (cannot find dll)
# use cmake . -DCMAKE_LIBRARY_PATH=*** to find library
cmake_minimum_required(VERSION 3.20)  # cmake version
set(CMAKE_CXX_STANDARD 17)  # c++ standard

project(CMAKE_TEST)  # project name
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)  # executable output path


include_directories(${PROJECT_SOURCE_DIR}/inc)  # add head file dir
find_library (libshare cmake_test_lib_share)
add_executable(main_link_shared main.c)  # build executable
target_link_libraries(main_link_shared ${libshare})  # link after build executable
]]
```

---

## link library

### 1. link_directories
使用`link_directories(${lib path})`，会自动在lib path中搜寻目标库

### 2. CMAKE_LIBRARY_PATH/CMAKE_PREFIX_PATH
编译命令指定搜寻目录，其中CMAKE_PREFIX_PATH会先搜寻该目录，再搜寻CMAKE_PREFIX_PATH/lib 目录
> cmake . -DCMAKE_LIBRARY_PATH=***

### 3. find_package
[a good guide](https://blog.csdn.net/zhanghm1995/article/details/105466372)
find_package 基于 `lib${lib name}Config.cmake` 文件搜索头文件和库，指定搜索三方库目录的方式比较多，推荐使用指定环境变量 `{lib name}_DIR` 的方式，在工程cmake调用 find_package 接口进行包含即可：
```
find_package(onnxruntime REQUIRED)
```
搜索成功后，相应的头文件和库集合变量可以查询 `lib${lib name}Config.cmake`  中的target寻找，一般不用再次手动包含。

### 4. pkg-config
[a good guide](https://blog.csdn.net/qq_21438461/article/details/132898233)
pkg-config 基于 pc 文件搜索头文件和库，在工程cmake中使用如下代码搜索
```
find_package(PkgConfig REQUIRED)  # 找到PkgConfig 包
pkg_search_module(DEFINE_NAME REQUIRED ${lib name})   # 使用PkgConfig 寻找库
```
这里，lib name 要和pc文件中的 Name 变量一致。成功匹配后，可使用 ${DEFINE_NAME}_INCLUDE_DIRS、${DEFINE_NAME}_LIBRARIES 两个变量，分别是头文件和库集合，来源于pc文件中的 Cflags 和 Libs 变量。
三方库的pc文件，可指定环境变量 `export PKG_CONFIG_PATH=${pc path}` 来搜寻。