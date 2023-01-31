#! /home/a/anaconda3/bin/python
# -*- coding: UTF-8 -*-
import multiprocessing
import os
from abc import ABC
import tornado
from tornado.httpserver import HTTPServer
import tornado.ioloop
import tornado.web
import tornado.log
from tornado.options import options
import logging
access_logger = logging.getLogger("tornado.access")

try:
    from make_app import make_app
except ImportError:
    class HelloWorldHandler(tornado.web.RequestHandler, ABC):
        def get(self):
            self.write("hello world!")

    def make_app():
        return tornado.web.Application([
            (r"/", HelloWorldHandler)
        ])

try:
    from make_app import num_processes
except ImportError:
    num_processes = multiprocessing.cpu_count()

try:
    from make_app import on_process_start
except ImportError:
    def on_process_start():
        access_logger.info("default on_process_start at: {}".format(os.getpid()))


# 格式化日志输出格式
# 默认是这种的：[I 160807 09:27:17 web:1971] 200 GET / (::1) 7.00ms
# 格式化成这种的：[2016-08-07 09:38:01 执行文件名:执行函数名:执行行数 日志等级] 内容消息
class LogFormatter(tornado.log.LogFormatter):
    def __init__(self):
        super(LogFormatter, self).__init__(
            fmt='%(color)s[%(asctime)s %(filename)s:%(funcName)s:%(lineno)d %(levelname)s]%(end_color)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


if __name__ == "__main__":
    tornado.options.define("port", default="8888", help="run on the port", type=int)  # 设置全局变量port
    tornado.options.parse_command_line()  # 启动应用前面的设置项目
    [i.setFormatter(LogFormatter()) for i in logging.getLogger().handlers]
    app = make_app()
    server = HTTPServer(app, max_header_size=int(1e10), max_body_size=int(1e10))
    server.listen(tornado.options.options.port)
    access_logger.info("listening on port: {port}, num_processes: {num_processes}".format(
        port=tornado.options.options.port,
        num_processes=num_processes
    ))
    server.start(num_processes=num_processes)
    main_loop = tornado.ioloop.IOLoop.current()
    main_loop.add_callback(on_process_start)
    main_loop.start()
