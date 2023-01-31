#!/bin/bash
APP_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/..; pwd)
if [ -z "$APP_NAME" ]; then
        APP_NAME=$(basename "${APP_HOME}")
fi
cp -rf $APP_HOME/target/$APP_NAME/src/* /home/service/ || exit 1
/home/service/run_xflow_service.sh 7001