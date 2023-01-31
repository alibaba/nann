#!/bin/bash

APP_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/..; pwd)
source "$APP_HOME/bin/setenv.sh"

HTTP_PORT="80"
CURL_BIN=/usr/bin/curl
HTTP_IP="http://$(hostname -i):$HTTP_PORT"

#####################################
checkpage() {

    self_check=
    ## using custom preload.sh instead
    if [[ -f ${APP_HOME}/target/${APP_NAME}/bin/preload.sh ]]; then
        echo "Run custom preload.sh for self-check"
        sh ${APP_HOME}/target/${APP_NAME}/bin/preload.sh
        self_check=$?

    ## using custom appctl.sh instead
    elif [[ -f "${APP_HOME}/target/${APP_NAME}/bin/appctl.sh" ]]; then
        echo "Run custom appctl.sh for self-check"
        sh ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh status
        self_check=$?

    ## loading given page for self-check
    else
        URL=$1; TITLE=$2; CHECK_TXT=$3
        if [ "$TITLE" == "" ]; then
          TITLE=$URL
        fi
        self_check=1
        echo "Load page for self-check -- ${HTTP_IP}${URL}"
        TMP_FILE=`$CURL_BIN -m 150 "${HTTP_IP}${URL}" 2>&1`
        (echo "$TMP_FILE" | fgrep "$CHECK_TXT") && self_check=0
    fi

    ## check status of self_check
    if [[ "$self_check" = "0" ]]; then
        status=1; error=0;
        echo "self-check status [  OK  ]"
    else
        status=0; error=1;
        echo "self-check status [FAILED]"
    fi
    return $error
}
#####################################
checkpage "/checkpreload.htm" "app" "success"