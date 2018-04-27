cmake_minimum_required (VERSION 3.6)
project (ProgParallel)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules")

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)
find_package(benchmark REQUIRED)
find_package(TBB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-W -Wall -mavx2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -s")

include_directories(lib/include)

add_library(renderer src/render.hpp src/render.cpp)
add_executable(view src/view.cpp)
#target_include_directories(view PRIVATE ${SDL2_INCLUDE_DIRS})
target_include_directories(view PRIVATE "/usr/include/SDL2")
target_link_libraries(view renderer ${SDL2_LIBRARIES} Threads::Threads)
#target_link_libraries(renderer tbb)

add_executable(bench src/bench.cpp)
target_link_libraries(bench renderer benchmark::benchmark)

