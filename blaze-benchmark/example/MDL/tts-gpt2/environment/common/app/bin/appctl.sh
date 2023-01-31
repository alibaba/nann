#!/bin/bash

PROG_NAME=$0
ACTION=$1
ALIYUN_MIRROR="http://yum.tbsite.net/pypi/simple/"

usage() {
    echo "Usage: $PROG_NAME {start|stop|restart|deploy|status|kill}"
    exit 1;
}

if [ "$UID" -eq 0 ]; then
    echo "can't run as root, please use: sudo -u admin $0 $@"
    exit 1
fi

APP_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/..; pwd)
VIRTUALENV_HOME="${APP_HOME}/virtualenv"
source "$APP_HOME/bin/setenv.sh"
source "$APP_HOME/bin/hook.sh"

if [ $# -lt 1 ]; then
    usage
fi

shift
TARGET=$*

die() {
    if [ "$#" -gt 0 ]; then
        echo "ERROR:" "$@"
    fi
    exit 128
}

up_virtualenv(){
    if [ -f ${VIRTUALENV_HOME}/virtual.rev ]; then
	local now_rev=`rpm -qv tops-python27-virtualenv`
        local rec_rev=`cat ${VIRTUALENV_HOME}/virtual.rev`
        if [[ "$now_rev" != "${rec_rev}" ]];then
            echo "[info]: update tops-python27-virtualenv. Begin rm -rf $VIRTUALENV_HOME"
            rm -rf ${VIRTUALENV_HOME}
        fi
    else
	rpm -qv tops-python27-virtualenv > ${VIRTUALENV_HOME}/virtual.rev
    fi
}

touch_virtualenv() {
    up_virtualenv
    if [ -d ${VIRTUALENV_HOME} ]; then
        echo "virtualenv exists, skip"
    else
        virtualenv -p python2.7 ${VIRTUALENV_HOME}
        if [ "$?" = 0 ]; then
            echo "INFO: create virtualenv success"
        else
            echo "ERROR: create virtualenv failed"
            exit 1
        fi
	rpm -qv tops-python27-virtualenv > ${VIRTUALENV_HOME}/virtual.rev
    fi
    echo "INFO: activate virtualenv..."
    source ${VIRTUALENV_HOME}/bin/activate || exit 1
}

extract_tgz() {
    local tgz_path="$1"
    local dir_path="$2"

    echo "extract ${tgz_path}"
    cd "${APP_HOME}/target" || exit 1
    rm -rf "${dir_path}" || exit 1
    tar xzf "${tgz_path}" || exit 1
    test -d "${dir_path}" || die "ERROR: no directory: ${dir_path}"
    touch --reference "${tgz_path}" "${tgz_path}.timestamp" || exit 1
}

update_target() {
    local tgz_name="$1"
    local dir_name="$2"

    local tgz_path="${APP_HOME}/target/${tgz_name}"
    local dir_path="${APP_HOME}/target/${dir_name}"

    local error=0
    # dir exists
    if [ -d "${dir_path}" ]; then
        # tgz exists
        if [ -f "${tgz_path}" ]; then
            local need_tar=0
            if [ ! -e "${tgz_path}.timestamp" ]; then
                need_tar=1
            else
                local tgz_time=$(stat -L -c "%Y" "${tgz_path}")
                local last_time=$(stat -L -c "%Y" "${tgz_path}.timestamp")
                if [ $tgz_time -gt $last_time ]; then
                    need_tar=1
                fi
            fi
            # tgz is new - extract_tgz
            if [ "${need_tar}" -eq 1 ]; then
                extract_tgz "${tgz_path}" "${dir_path}"
            fi
            # tgz is not new - return SUCCESS
        fi
        # tgz not exists - return SUCCESS
    # dir not exists
    else
        # tgz exists - extract_tgz
        if [ -f "${tgz_path}" ]; then
            extract_tgz "${tgz_path}" "${dir_path}"
        # tgz not exists - return FAIL
        else
            echo "ERROR: ${tgz_path} NOT EXISTS"
            error=1
        fi
    fi

    return $error
}

check_requirements() {
    touch_virtualenv
    echo "INFO: begin install requirements..."
    if [ -f ${APP_HOME}/target/${APP_NAME}/requirements.txt ]; then
        cp -f ${APP_HOME}/target/${APP_NAME}/requirements.txt ${APP_HOME}/conf/requirements.txt || exit 1
    fi
    # check python requirements
    if ! [ -f "${APP_HOME}/conf/requirements.txt" ]; then
        echo "ERROR: app requirements not found, it's rarely that an app doesn't have any dependencies"
        echo "if you confirm this is right, touch an empty requirements.txt in ${APP_HOME}/conf to avoid this error"
        exit
    fi
    if ! [ -d ${APP_HOME}/logs/app/ ]; then
        mkdir -p ${APP_HOME}/logs/app/ || exit 1
    fi
    local requirements_log="${APP_HOME}/logs/app/${APP_NAME}_requirements.log"
    touch "$requirements_log" || exit
    if [ `cat ${APP_HOME}/conf/requirements.txt | wc -l` -ne 0 ]; then
        pip install -r "${APP_HOME}/conf/requirements.txt" -i "${ALIYUN_MIRROR}" |tee -a "${requirements_log}" || exit 1
        # wait
        local pip_res=$?
        if [ $pip_res -ne 0 ]; then
            echo "ERROR: requirements not satisfied and auto install failed, please check ${requirements_log}"
            exit 1
        fi
    fi

    # 增加一个安装本地源的 技术保障安全有需要
    if [ -f ${APP_HOME}/target/${APP_NAME}/pypi/requirements.txt ]; then
        pip install --no-index --find-link=file://${APP_HOME}/target/${APP_NAME}/pypi/ -r ${APP_HOME}/target/${APP_NAME}/pypi/requirements.txt |tee -a "${requirements_log}" || exit 1

        local pip_lres=$?
        if [ $pip_lres -ne 0 ]; then
            echo "ERROR: requirements not satisfied and auto install failed, please check ${requirements_log}"
            exit 1
        fi
    fi
}

replace_command() {
    local start_linum=$(grep -n -m 1 "\[program:" ${APP_HOME}/conf/supervisord.conf | awk -F: '{print $1}')
    local command=$*
    sed -i "${start_linum}, +5s|command=.*|command=${command}|g" ${APP_HOME}/conf/supervisord.conf
}

determine_app_type() {

    if [ -r "${APP_HOME}/target/${APP_NAME}/uwsgi.ini" ]; then
        echo "uswgi app detected"
        local command="uwsgi --ini ${APP_HOME}/target/${APP_NAME}/uwsgi.ini --catch-exceptions --protocol=http"
        replace_command ${command}
        return 0
    fi

    if [ -r "${APP_HOME}/target/${APP_NAME}/gunicorn.py" ]; then
        echo "gunicorn app detected"
        local command="gunicorn --config ${APP_HOME}/target/${APP_NAME}/gunicorn.py"
        replace_command ${command}
        return 0
    fi

    if [ -r "${APP_HOME}/target/${APP_NAME}/start_cmd" ]; then
        echo "custom app detected"
        local command=$(cat ${APP_HOME}/target/${APP_NAME}/start_cmd)
        replace_command ${command}
        return 0
    fi

    if [ -r "${APP_HOME}/start_cmd" ]; then
        echo "custom app detected"
        local command=$(cat ${APP_HOME}/start_cmd)
        replace_command ${command}
        return 0
    fi

    echo "ERROR: cannot determine app type"
    echo "you must put one of [uwsgi.ini, gunicorn.ini, start_cmd or bin/appctl.sh|bin/preload.sh] file in ${APP_HOME}/target/${APP_NAME}"
    exit 1
}

check_supervisor_conf() {
    # check supervisord config files
    if ! [ -f "${APP_HOME}/conf/supervisord.conf" ]; then
        echo "ERROR: you must create a supervisord config for your app to make it control with supervisord"
        exit 1
    fi
    sed -i 's/{{app_name}}/'$(echo $APP_NAME)'/g' ${APP_HOME}/conf/supervisord.conf
}

kill_supervisord() {
    if { test -r "${SUPERVISOR_PID}" && kill -0 "$(cat "${SUPERVISOR_PID}")"; }; then
        kill $(cat "${SUPERVISOR_PID}")
    else
        echo "supervisord not running, do nothing"
    fi
}

start() {

    # delete old $SUPERVISOR_LOG, keep last 20 logs

    ls "$SUPERVISOR_LOG".* 2>/dev/null | tail -n +$((20 + 1)) | xargs --no-run-if-empty rm -f
    if [ -e "$SUPERVISOR_LOG" ]; then
        mv "$SUPERVISOR_LOG" "$SUPERVISOR_LOG.$(date '+%Y%m%d%H%M%S')" || exit 1
    fi
    mkdir -p "$(dirname "${SUPERVISOR_LOG}")" || exit 1
    touch "$SUPERVISOR_LOG" || exit 1

    # show locale
    locale >> "${SUPERVISOR_LOG}"

    # print start info to both ${SUPERVISOR_LOG} and console.
    do_start | tee -a "${SUPERVISOR_LOG}"
    status
    echo "apps started, use 'appctl.sh status' to check apps status later"
    if [ "${NGINX_SKIP}" -ne "1" ]; then
        echo "start nginx"
        sh "$NGINXCTL" start
    fi
}

do_start() {
    mkdir -p "${APP_HOME}/target" || exit
    mkdir -p "${APP_HOME}/logs" || exit
    mkdir -p "${APP_HOME}/logs/supervisord" || exit
    mkdir -p "${APP_HOME}/logs/app" || exit

    # update app
    update_target "${APP_NAME}.tgz" "${APP_NAME}" || exit 1

    # check requirements
    #echo "check requirements"
    #check_requirements

    # 如果应用有自定义启动脚本, 那么直接执行应用的脚本
    if [ -r ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]; then
        sh ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh start || exit 1
        return $?
    fi

    # check supervisord conf
    check_supervisor_conf

    # 应用启动脚本
    determine_app_type

    echo "start apps"
    beforeStartApp

    # start supervisor
    if ! { test -r "${SUPERVISOR_PID}" && kill -0 "$(cat "${SUPERVISOR_PID}")"; }; then
        supervisord -c "${APP_HOME}/conf/supervisord.conf" || exit 1
    else
        if [ -n "${TARGET}" ]; then
            supervisorctl -c "${APP_HOME}/conf/supervisord.conf" start "${TARGET}" || exit 1
        else
            supervisorctl -c "${APP_HOME}/conf/supervisord.conf" start all || exit 1
        fi
    fi
    kill -0 "$(cat ${SUPERVISOR_PID})" || exit 1
}

online() {
    # 如果应用有自定义启动脚本, 那么直接执行应用的脚本
    if [ -r ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]; then
        sh ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh online || exit 1
        return $?
    fi
    touch -m $STATUSROOT_HOME/status.taobao || exit 1
    echo "app auto online..."
}


offline() {
    # 如果应用有自定义启动脚本, 那么直接执行应用的脚本
    if [ -r ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]; then
        sh ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh offline || exit 1
        return $?
    fi
    rm -f $STATUSROOT_HOME/status.taobao || exit 1
    echo "wait app offline..."
    for e in $(seq 15); do
        echo -n " $e"
        sleep 1
    done
    echo
}

stop() {
    if [ "${NGINX_SKIP}" -ne "1" ]; then
        echo "stop nginx"
        sh "$NGINXCTL" stop
    fi
    echo "stop apps"
    beforeStopApp

    # 如果应用有自定义启动脚本, 那么直接执行应用的脚本
    if [ -r ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]; then
        sh ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh stop || exit 1
        return $?
    fi

    if [ -n "${TARGET}" ]; then
        supervisorctl -c "${APP_HOME}/conf/supervisord.conf" stop "${TARGET}"
    else
        supervisorctl -c "${APP_HOME}/conf/supervisord.conf" stop all
    fi
    afterStopApp
    echo "stop supervisord"
    kill_supervisord
}

status() {
    # 如果应用有自定义启动脚本, 那么直接执行应用的脚本
    if [ -r ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]; then
        sh ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh status || exit 1
        return $?
    fi
    supervisorctl -c "${APP_HOME}/conf/supervisord.conf" status
}

backup() {
    if [ -f "${APP_HOME}/target/${APP_NAME}.tgz" ]; then
        mkdir -p "${APP_HOME}/target/backup" || exit
        tgz_time=$(date --reference "${APP_HOME}/target/${APP_NAME}.tgz" +"%Y%m%d%H%M%S")
        cp -f "${APP_HOME}/target/${APP_NAME}.tgz" "${APP_HOME}/target/backup/${APP_NAME}.${tgz_time}.tgz"
    fi
}

new_app(){
    if [[ ! -d ${APP_HOME}/target/${APP_NAME} && -f ${APP_HOME}/target/${APP_NAME}.tgz ]]; then
        update_target "${APP_NAME}.tgz" "${APP_NAME}" || exit 1
    fi
}

new_app

case "$ACTION" in
    start)
        start
    ;;
    stop)
        stop
    ;;
    restart)
        stop
        start
    ;;
    pubstart)
        stop
        start
    ;;
    deploy)
        stop
        start
        backup
    ;;
    status)
        status
    ;;
    kill)
        kill_supervisord
    ;;
    online)
        online
    ;;
    offline)
        offline
    ;;
    *)
        usage
    ;;
esac