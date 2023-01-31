file(GLOB_RECURSE SOURCES
        "*.cc"
        "*.h")
add_library (user_ops STATIC ${SOURCES})
