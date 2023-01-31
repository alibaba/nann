file(GLOB_RECURSE SOURCE
        "common/*.cc"
        "core/*.cc")

set(proto_gen_folder "${PROJECT_BINARY_DIR}/benchmark/proto")
file(GLOB_RECURSE proto_files "proto/*.proto")
protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python
        "${PROJECT_SOURCE_DIR}/benchmark" "proto" ${proto_files})
list(APPEND SOURCE ${proto_srcs})
MESSAGE(${proto_srcs})

add_executable(benchmark ${SOURCE})

target_link_libraries(benchmark
	tensorflow_cc
        tensorflow_framework
        cppmetrics
        boost_chrono
        boost_system
        boost_thread
        glog
        pthread)
if (USE_CUDA)
    target_link_libraries(benchmark cudart)
endif ()
if (AIOS)
    SET(USER_OPS_WRAPPED -Wl,--whole-archive user_ops -Wl,--no-whole-archive)
    target_link_libraries(benchmark ${USER_OPS_WRAPPED})
endif ()
