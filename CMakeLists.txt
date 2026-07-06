cmake_minimum_required(VERSION 3.24)

project(
  sparse_voxel_octree
  VERSION 0.1.0
  DESCRIPTION "Sparse voxel octree C++/CUDA library with Python bindings"
  LANGUAGES CXX
)

include(GNUInstallDirs)

option(SVO_ENABLE_CUDA "Enable CUDA components" OFF)
option(SVO_ENABLE_PYTHON "Enable Python packaging hooks" OFF)
option(SVO_ENABLE_TORCH "Enable PyTorch integration hooks" OFF)
option(SVO_BUILD_TESTS "Build C++ smoke tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SVO_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(WARNING "SVO_ENABLE_CUDA=ON but no CUDA compiler was found; disabling CUDA targets.")
    set(SVO_ENABLE_CUDA OFF)
  endif()
endif()

add_library(svo_core STATIC src/Builder.cpp src/Camera.cpp src/Octree.cpp src/Query.cpp src/Raycast.cpp src/Version.cpp)
add_library(svo::core ALIAS svo_core)

target_include_directories(
  svo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(svo_core PUBLIC cxx_std_20)
set_target_properties(svo_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(
  svo_core
  PUBLIC
    SVO_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    SVO_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    SVO_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    SVO_ENABLE_CUDA=$<BOOL:${SVO_ENABLE_CUDA}>
)

if(SVO_ENABLE_CUDA)
  target_sources(svo_core PRIVATE cuda/generate_rays.cu cuda/query_points.cu cuda/raycast.cu)
  set_target_properties(
    svo_core
    PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
  )
  target_link_libraries(svo_core PUBLIC CUDA::cudart)

  add_executable(svo_point_query_benchmark benchmarks/cpp/point_query_benchmark.cpp)
  target_link_libraries(svo_point_query_benchmark PRIVATE svo_core)

  add_executable(svo_raycast_benchmark benchmarks/cpp/raycast_benchmark.cpp)
  target_link_libraries(svo_raycast_benchmark PRIVATE svo_core)
endif()

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

if(SVO_ENABLE_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_svo MODULE python/bindings.cpp)
  target_link_libraries(_svo PRIVATE svo_core)
  target_include_directories(
    _svo
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
  )
  target_compile_features(_svo PRIVATE cxx_std_20)

  install(TARGETS _svo LIBRARY DESTINATION svo RUNTIME DESTINATION svo)
endif()

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

  add_executable(svo_raycast_test tests/cpp/test_raycast.cpp)
  target_link_libraries(svo_raycast_test PRIVATE svo_core)
  add_test(NAME svo_raycast_test COMMAND svo_raycast_test)

  add_executable(svo_camera_test tests/cpp/test_camera.cpp)
  target_link_libraries(svo_camera_test PRIVATE svo_core)
  add_test(NAME svo_camera_test COMMAND svo_camera_test)

  add_executable(svo_device_buffer_test tests/cpp/test_device_buffer.cpp)
  target_link_libraries(svo_device_buffer_test PRIVATE svo_core)
  add_test(NAME svo_device_buffer_test COMMAND svo_device_buffer_test)

  add_executable(svo_public_headers_test tests/cpp/test_public_headers.cpp)
  target_link_libraries(svo_public_headers_test PRIVATE svo_core)
  add_test(NAME svo_public_headers_test COMMAND svo_public_headers_test)

  if(SVO_ENABLE_CUDA)
    add_executable(svo_query_cuda_test tests/cpp/test_query_cuda.cpp)
    target_link_libraries(svo_query_cuda_test PRIVATE svo_core)
    add_test(NAME svo_query_cuda_test COMMAND svo_query_cuda_test)

    add_executable(svo_raycast_cuda_test tests/cpp/test_raycast_cuda.cpp)
    target_link_libraries(svo_raycast_cuda_test PRIVATE svo_core)
    add_test(NAME svo_raycast_cuda_test COMMAND svo_raycast_cuda_test)

    add_executable(svo_camera_cuda_test tests/cpp/test_camera_cuda.cpp)
    target_link_libraries(svo_camera_cuda_test PRIVATE svo_core)
    add_test(NAME svo_camera_cuda_test COMMAND svo_camera_cuda_test)
  endif()
endif()
