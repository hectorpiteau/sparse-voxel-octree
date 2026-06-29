cmake_minimum_required(VERSION 3.24)

project(
  sparse_voxel_octree
  VERSION 0.1.0
  DESCRIPTION "Sparse voxel octree C++/CUDA library with Python bindings"
  LANGUAGES CXX
)

option(SVO_ENABLE_CUDA "Enable CUDA components" OFF)
option(SVO_ENABLE_PYTHON "Enable Python packaging hooks" ON)
option(SVO_ENABLE_TORCH "Enable PyTorch integration hooks" OFF)
option(SVO_BUILD_TESTS "Build C++ smoke tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SVO_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "SVO_ENABLE_CUDA=ON but no CUDA compiler was found; disabling CUDA targets.")
    set(SVO_ENABLE_CUDA OFF)
  endif()
endif()

add_library(svo_core STATIC src/Builder.cpp src/Octree.cpp src/Query.cpp src/Version.cpp)
add_library(svo::core ALIAS svo_core)

target_include_directories(
  svo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(svo_core PUBLIC cxx_std_17)
target_compile_definitions(
  svo_core
  PUBLIC
    SVO_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    SVO_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    SVO_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    SVO_ENABLE_CUDA=$<BOOL:${SVO_ENABLE_CUDA}>
)

install(
  TARGETS svo_core
  EXPORT svoTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/svo DESTINATION include)
install(DIRECTORY third_party/glm/glm DESTINATION include)
install(FILES third_party/glm/copying.txt DESTINATION share/licenses/glm)

install(
  EXPORT svoTargets
  NAMESPACE svo::
  DESTINATION lib/cmake/svo
)

include(CTest)

if(BUILD_TESTING AND SVO_BUILD_TESTS)
  add_executable(svo_smoke_test tests/cpp/test_smoke.cpp)
  target_link_libraries(svo_smoke_test PRIVATE svo_core)
  add_test(NAME svo_smoke_test COMMAND svo_smoke_test)

  add_executable(svo_builder_test tests/cpp/test_builder.cpp)
  target_link_libraries(svo_builder_test PRIVATE svo_core)
  add_test(NAME svo_builder_test COMMAND svo_builder_test)

  add_executable(svo_query_test tests/cpp/test_query.cpp)
  target_link_libraries(svo_query_test PRIVATE svo_core)
  add_test(NAME svo_query_test COMMAND svo_query_test)

  add_executable(svo_public_headers_test tests/cpp/test_public_headers.cpp)
  target_link_libraries(svo_public_headers_test PRIVATE svo_core)
  add_test(NAME svo_public_headers_test COMMAND svo_public_headers_test)
endif()
