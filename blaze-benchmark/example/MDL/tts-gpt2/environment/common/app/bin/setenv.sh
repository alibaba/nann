# app
# set ${APP_NAME}, if empty $(basename "${APP_HOME}") will be used.
APP_NAME=
NGINX_HOME=/home/admin/cai

# os env
# NOTE: must edit LANG and JAVA_FILE_ENCODING together
export LANG=zh_CN.UTF-8
export NLS_LANG=AMERICAN_AMERICA.ZHS16GBK
export LD_LIBRARY_PATH=/opt/taobao/oracle/lib:/opt/taobao/lib:$LD_LIBRARY_PATH
export PATH=/home/tops/bin:$PATH
export CPU_COUNT="$(grep -c 'cpu[0-9][0-9]*' /proc/stat)"
ulimit -c unlimited

# if set to "1", skip start nginx.
test -z "$NGINX_SKIP" && NGINX_SKIP=0

# set port for checking status.taobao file. Comment it if no need.
STATUS_PORT=80

# env check and calculate
#
if [ -z "$APP_NAME" ]; then
        APP_NAME=$(basename "${APP_HOME}")
fi
if [ -z "$NGINX_HOME" ]; then
        NGINX_HOME=/home/admin/cai
fi

export PYTHONPATH=$PYTHONPATH:${APP_HOME}/target/${APP_NAME}

SUPERVISOR_LOG="${APP_HOME}/logs/supervisord/supervisord.log"
#STATUSROOT_HOME="${APP_HOME}/target/${APP_NAME}.war"
STATUSROOT_HOME="${NGINX_HOME}/htdocs"
NGINXCTL=$NGINX_HOME/bin/nginxctl

SUPERVISOR_PID="/home/admin/logs/supervisord.pid"