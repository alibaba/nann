FROM reg.docker.alibaba-inc.com/blaze/jinluyang.jly:TRT8opensource_model_20211020_bothuse_TRT_kvcache_fp16

ENV APP_NAME get-pict-label


## 构建时要做的事，一般是执行shell命令，例如用来安装必要软件，创建文件（夹），修改文件
#RUN rpm -ivh --nodeps "http://yum.corp.taobao.com/taobao/7/x86_64/current/c_tbip/c_tbip-2.0.9-1580249.18bc121.el7.x86_64.rpm" && \
#    rpm -ivh --nodeps "http://yum.corp.taobao.com/taobao/6/x86_64/stable/taobao-cronolog/taobao-cronolog-1.7.0b-2.el6.x86_64.rpm" && \
#    rpm -ivh --nodeps "http://yum.tbsite.net/taobao/7/x86_64/current/tengine-proxy/tengine-proxy-2.1.16-20180517093604.el7u2.x86_64.rpm"

RUN mkdir -p /home/admin/$APP_NAME/bin && \
mkdir -p /home/admin/$APP_NAME/conf && \
mkdir -p /home/admin/$APP_NAME/target && \
mkdir -p /home/admin/$APP_NAME/dependencies && \
echo "sudo /home/a/vipsrv-dns/bin/vipsrv enable" >> /home/admin/start.sh && \
echo "sudo /home/a/vipsrv-dns/bin/vipsrv restart" >> /home/admin/start.sh && \
echo "sudo chown -R admin:admin /home/service/" >> /home/admin/start.sh && \
echo "export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2" >> /home/admin/start.sh && \
echo "export PATH=$PATH:/home/a/anaconda3/bin/" >> /home/admin/start.sh && \
echo "/home/admin/$APP_NAME/bin/appctl.sh pubstart" >> /home/admin/start.sh && \
echo "/home/admin/$APP_NAME/bin/appctl.sh stop" > /home/admin/stop.sh && \
echo "/home/admin/$APP_NAME/bin/preload.sh" >> /home/admin/health.sh && \
echo "/home/admin/$APP_NAME/bin/run.sh" > /home/admin/$APP_NAME/start_cmd

# 将应用启动脚本和配置文件复制到镜像中
COPY environment/common/app/ /home/admin/${APP_NAME}/
COPY environment/common/cai/ /home/admin/cai/

# 设置文件操作权限
RUN chmod -R a+x /home/admin/$APP_NAME/bin/ && chmod +x /home/admin/*.sh

# 安装依赖
#RUN /home/a/anaconda3/bin/pip3 install -i https://pypi.antfin-inc.com/simple/ -r /home/admin/${APP_NAME}/conf/requirements.txt
#RUN /home/a/anaconda3/bin/pip3 install -i https://pypi.antfin-inc.com/simple/ torchvision opencv-python
RUN /home/a/anaconda3/bin/pip3 --default-timeout=100 --no-cache-dir install git+http://gitlab-ci-token:cc138c7ff2a153dab50de0d2256f91@gitlab.alibaba-inc.com/jinluyang.jly/ToT.git@master
#RUN pip3 install -i https://pypi.antfin-inc.com/simple/ -r /home/admin/${APP_NAME}/conf/requirements.txt && yum -y install -b current jemalloc-4.0.4-866559.alios7.x86_64

#RUN pip3 --no-cache-dir install -i https://pypi.antfin-inc.com/simple/ git+http://gitlab-ci-token:0eb6a083efa60bf41b73f89e45b4e1@gitlab.alibaba-inc.com/vodl/istio_pyservice_utils.git@0.0.7

# 启动容器时进入的工作目录
WORKDIR /home/admin/$APP_NAME/bin

#容器启动时自动执行的脚本，我们一般会将应用启动脚本放在这里，相当于系统自启应用
#ENTRYPOINT ["/home/admin/start.sh"]

# 将构建出的主包复制到指定镜像目录中
#COPY $APP_NAME.tgz /home/admin/$APP_NAME/target/
COPY src /home/service

#RUN chown admin:admin -R /home/admin

#RUN echo "kernel.core_pattern=/home/admin/rttm/core_%e_%p" >> /etc/sysctl.d/rttm.conf && echo "kernel.core_uses_pid=0" >> /etc/sysctl.d/rttm.conf
