cmake_minimum_required(VERSION 3.6)
project(BlazeBenchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(USE_CUDA "use cuda" ON)
option(AIOS "use internal tf branch" OFF)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS_DEBUG "-g")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
endif ()
set(CMAKE_BUILD_TYPE RELEASE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=parentheses -Wno-error=unused-local-typedefs -Wno-error=deprecated-declarations")
if (AIOS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif ()
if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "5")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=placement-new=")
endif ()

# CUDA
if (USE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGOOGLE_CUDA=1")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
    find_package(CUDA REQUIRED)
    include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
    include(${PROJECT_SOURCE_DIR}/cmake/Cuda.cmake)
    include_directories(
            ${CUDA_PATH}/include/
            ${CUDNN_ROOT_DIR}/include
    )
    link_directories(
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/
            ${CUDNN_ROOT_DIR}/lib64/
    )
endif ()

# BOOST
SET(BOOST_INC_DIR "${PROJECT_SOURCE_DIR}/thirdparty/boost_1_53_0" ${CMAKE_INCLUDE_PATH})
SET(BOOST_LIB_DIR "${PROJECT_SOURCE_DIR}/thirdparty/boost_1_53_0/stage/lib" ${CMAKE_LIBRARY_PATH})

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/thirdparty/cppmetrics/src
        ${BOOST_INC_DIR}
)
link_directories(
        ${BOOST_LIB_DIR}
)

if (AIOS)
    include_directories(
            ${AIOS_SRC_DIR}/bazel-aios/bazel-out/host/bin/external/org_tensorflow
            ${AIOS_SRC_DIR}/bazel-aios/external/org_tensorflow
            ${AIOS_SRC_DIR}/bazel-aios/external/eigen_archive/
            ${AIOS_SRC_DIR}/bazel-aios/external/com_google_absl
            ${AIOS_SRC_DIR}/bazel-aios/external/flatbuffers/include/
            ${AIOS_SRC_DIR}/bazel-aios/external/com_google_protobuf/src
    )
    link_directories(
            ${AIOS_SRC_DIR}/bazel-bin/external/org_tensorflow/tensorflow/
    )
else ()
    include_directories(
            ${TF_SRC_DIR}
            ${TF_SRC_DIR}/bazel-genfiles/
            ${TF_SRC_DIR}/bazel-tensorflow/external/eigen_archive/
            ${TF_SRC_DIR}/bazel-tensorflow/external/com_google_absl
            ${TF_SRC_DIR}/bazel-tensorflow/external/flatbuffers/include/
            ${TF_SRC_DIR}/bazel-tensorflow/external/com_google_protobuf/src
    )
    link_directories(
            ${TF_SRC_DIR}/bazel-bin/tensorflow/
    )
endif ()

# set protobuf
if (AIOS)
    set(PROTOBUF_PROTOC_EXECUTABLE ${AIOS_SRC_DIR}/bazel-aios/bazel-out/host/bin/external/com_google_protobuf/protoc)
else ()
    set(PROTOBUF_PROTOC_EXECUTABLE ${TF_SRC_DIR}/bazel-tensorflow/bazel-out/host/bin/external/com_google_protobuf/protoc)
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/ProtoBuf.cmake)

add_subdirectory(thirdparty)
add_subdirectory(benchmark)
if (AIOS)
    add_subdirectory(user_ops)
endif ()
