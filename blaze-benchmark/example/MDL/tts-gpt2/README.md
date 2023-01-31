# 运行方式
容器、镜像的名字,挂载的目录改成自己的)：
```bash
sudo docker build -t jly:test1 .
sudo nvidia-docker run --net=host --name=jlytestdockerfile -it -v /home/jinluyang.jly/:/home/jinluyang.jly/ jly:test1 /bin/bash
sudo ldconfig
cd /home/service
PYTHONPATH=. python rttm/modules/generate_design_points_text.py
```
可以生成亮点文案。需要网络可以访问商品中心  

# 性能数据
[持续进行的TTS服务性能优化，包括两个模型，其中第二个是gpt-2](https://yuque.antfin-inc.com/docs/share/77e8daa7-60e2-471d-b50b-e8980c7aee2d?#IXQf 《TTS服务优化效果》)

gpt-2使用TensorRT+kvcache优化，在Tesla T4上每个loop时间约为8ms，128个loop总时间约为1s，在上述链接中对应27行D列
