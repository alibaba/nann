#!/bin/sh
srcdir=`pwd`

#脚本内置函数get_tgz可以直接引用。会把源码目录的打成压缩包。并且放到指定目录，并且生成 md5 文件
#get_tgz 第一个参数是包名，第二个参数是需要进入的目录，第三个参数是要打成压缩包的目标目录或文件
get_tgz "${APP_NAME}____${SCHEMA_NAME}.tgz" "$srcdir/../" ${APP_NAME}

#结果检测
check_result "$?" "execute default build get_tgz function"

#信息打印
build_info "Package done!!!"
