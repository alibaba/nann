#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/10
# update: 2020/9/10
"""
make_app.py:
"""
from abc import ABC
import tornado.web
import json
import os
import time

import logging
util_logger = logging.getLogger("tornado.access")

from run import GenerateText, GetPictLabel
import sys
sys.path.append('high_service')
from kmonitor.reporter import StatusReporter
import subprocess

num_processes = 2

def start_mps():
    gpu_p100 = "P100"
    device_map = {gpu_p100 : "15f8", "T4" : "1eb8"}
    nvi = subprocess.Popen("/usr/sbin/lspci | grep -i nvidia", shell=True, stdout = subprocess.PIPE)
    out = nvi.communicate()[0].decode('utf-8')
    if device_map[gpu_p100] in out:
        logging.info("P100 found")
        return

    path_0 = "/usr/bin/nvidia-cuda-mps-control"
    path_1 = "/usr/local/nvidia/bin/nvidia-cuda-mps-control"
    prefix_1 = "PATH=/usr/local/nvidia/bin/ LD_LIBRARY_PATH=/usr/local/nvidia/lib64 "
    mps_path = ""
    if os.path.isfile(path_0):
        mps_path = path_0
    elif os.path.isfile(path_1):
        mps_path = prefix_1 + path_1
    else:
        logging.warn("Cannot find nvidia-cuda-mps-control.")
        return

    start_mps = mps_path + " -d"
    start_proc = subprocess.Popen(start_mps, shell=True)

def get_cluster_name():
    return "text-generation-design-point-opt"

def on_process_start():
    if tornado.process.task_id() == 0:
        start_mps()
    global reporter
    cluster_name = get_cluster_name()
    reporter = StatusReporter(cluster_name)
    if not os.path.exists("/usr/local/nvidia/lib64"):
        util_logger.info("/usr/local/nvidia/lib64 not mounted !!")
    util_logger.info("ddm on_process_start at %s" % os.getpid())
    global gt
    gt = GenerateText("conf/generate_design_points_text.json")
    global gpl
    gpl = GetPictLabel("conf/get_pict_label.json")


class RealTimeTextMakingSampleHandler(tornado.web.RequestHandler, ABC):
    def data_received(self, chunk):
        pass

    def get(self):
        self.write(json.dumps({"text": "Hello, RTTM!!!"}))


class RealTimeTextMakingDesignPointsHandler(tornado.web.RequestHandler, ABC):
    global reporter
    def prepare(self):
        self.start = time.time()
        self.error_msg = None

    def on_finish(self):
        if reporter:
            rt = time.time() - self.start
            reporter.report_rt(rt)
            reporter.report_error(self.error_msg)

    def data_received(self, chunk):
        pass

    def get(self):
        util_logger.info('[Raw Request]{}'.format(self.request))
        rst = {'status': False, 'error_msg': ''}

        # get query
        try:
            query_info = {
                'entity_id': self.get_query_argument("entity_id"),
                'prefix_flag': self.get_query_argument("prefix_flag", default=1),
                'extra_info': self.get_query_argument("extra_info", default={})
            }
            util_logger.info('[Parsed Request]{}'.format(query_info))
            entity_ids = query_info["entity_id"].split(",")
            prefix_flag = bool(int(query_info['prefix_flag']))
            text_list = gt.run(entity_ids, prefix_flag)

            data = []
            for entity_id, text in zip(entity_ids, text_list):
                text_data = {"design-point": text}
                data.append({"entity_id": entity_id,
                                "text": text_data})

            self._add_result_key(rst, "data", data)
            if len(data) == 0:
                self._add_result_key(rst, "error_msg", "data is empty!!!")
            else:
                self._add_result_key(rst, "status", True)

        except tornado.web.MissingArgumentError as e:
            rst['error_msg'] = 'Incorrect query. Necessary params: scene_id, product_id, entity_id, entity_type, ' \
                               'text_type, max_length, min_length, count, extra_info'
            rst["data"] = []
            util_logger.warn('[Incorrect query] error_msg is : {}'.format(e))
            self.error_msg = rst['error_msg']
        except Exception as e:
            rst['error_msg'] = str(e)
            rst["data"] = []
            util_logger.warn('[Exception]')
            util_logger.warn('task id:{}'.format(str(tornado.process.task_id())))
            self.error_msg = rst['error_msg']

        rst_str = json.dumps(rst, ensure_ascii=False)
        util_logger.info('[ProcessResult] Query is {}. result is {:.500}'.format(self.request, rst_str))
        self.write(rst_str)

    def _add_result_key(self, rst, key, value):
        '''
        add task-specific key into the result
        '''

        return rst.update({key: value})

class RealTimeTextMakingPictLabelsHandler(tornado.web.RequestHandler, ABC):
    def data_received(self, chunk):
        pass

    def get(self):
        util_logger.info('[Raw Request]{}'.format(self.request))
        rst = {'status': False, 'error_msg': ''}

        # get query
        try:
            query_info = {
                'entity_id': self.get_query_argument("entity_id"),
                'pict_url': self.get_query_argument("pict_url", default=None),
                'top_k': self.get_query_argument("top_k", default=15),
                'extra_info': self.get_query_argument("extra_info", default={})
            }
            util_logger.info('[Parsed Request]{}'.format(query_info))
            entity_ids = query_info["entity_id"].split(",")
            pict_url = query_info["pict_url"]
            top_k = int(query_info['top_k'])
            attr_list_list = gpl.run(entity_ids, top_k, pict_url)

            data = []
            for entity_id, attr_list in zip(entity_ids, attr_list_list):
                data.append({"entity_id": entity_id,
                                "attr_list": attr_list})

            self._add_result_key(rst, "data", data)
            if len(data) == 0:
                self._add_result_key(rst, "error_msg", "data is empty!!!")
            else:
                self._add_result_key(rst, "status", True)

        except tornado.web.MissingArgumentError as e:
            rst['error_msg'] = 'Incorrect query. Necessary params: scene_id, product_id, entity_id, entity_type, ' \
                               'text_type, max_length, min_length, count, extra_info'
            rst["data"] = []
            util_logger.warn('[Incorrect query] error_msg is : {}'.format(e))
        except Exception as e:
            rst['error_msg'] = str(e)
            rst["data"] = []
            util_logger.warn('[Exception]')

        rst_str = json.dumps(rst, ensure_ascii=False)
        util_logger.info('[ProcessResult] Query is {}. result is {:.500}'.format(self.request, rst_str))
        self.write(rst_str)

    def _add_result_key(self, rst, key, value):
        '''
        add task-specific key into the result
        '''

        return rst.update({key: value})

class HealthCheckHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("success")
    def _log(self):
        pass

def make_app():
    return tornado.web.Application([
        (r"/rttm/sample", RealTimeTextMakingSampleHandler),
        (r"/rttm/dp", RealTimeTextMakingDesignPointsHandler),
        (r"/rttm/pl", RealTimeTextMakingPictLabelsHandler),
        (r"/checkpreload.htm", HealthCheckHandler),
        (r"/status.taobao", HealthCheckHandler)
    ])
