#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/10
# update: 2020/9/10
"""
run.py.py:
"""
import requests
import rttm
from framework.pipeline import Pipeline
from framework.component import Message
import logging
import time


class GenerateText(object):
    def __init__(self, conf_path):
        self.pipeline = Pipeline(conf_path)

    def run(self, item_ids, prefix_flag, **kwargs):
        message_list = self.generate_message(item_ids, prefix_flag)
        result = self.pipeline.process(message_list)
        return self.post_process(result)

    def generate_message(self, item_ids, prefix_flag):
        message_list = []
        for item_id in item_ids:
            message_list.append(Message({
                "item_id": item_id,
                "prefix_flag": prefix_flag
            }))

        return message_list

    def post_process(self, result):
        text_list = [ele['text'] for ele in result]
        return text_list

class GetPictLabel(object):
    def __init__(self, conf_path):
        self.pipeline = Pipeline(conf_path)

    def run(self, item_ids, top_k, pict_url, **kwargs):
        message_list = self.generate_message(item_ids, top_k, pict_url)
        result = self.pipeline.process(message_list)
        return self.post_process(result)

    def generate_message(self, item_ids, top_k, pict_url):
        message_list = []
        for item_id in item_ids:
            message_list.append(Message({
                "item_id": item_id,
                "assigned_pict_url": pict_url,
                "top_k": top_k
            }))

        return message_list

    def post_process(self, result):
        text_list = [ele['attr_list'] for ele in result]
        return text_list


if __name__ == "__main__":
    gt = GenerateText("conf/generate_design_points_text.json")
    #entity_ids = "592772768984,625332353702".split(",")
    entity_ids = "592772768984".split(",")
    prefix_flag = False
    text_list = gt.run(entity_ids, prefix_flag)
    start = time.time()
    for _ in range(5):
      text_list = gt.run(entity_ids, prefix_flag)
    print('cost time ', (time.time() - start) / 5)
    logging.info("dooo")
    data = []
    title = '新款连衣裙'
    for entity_id, text in zip(entity_ids, text_list):
        data.append({"entity_id": entity_id,
                    "title": title})
    logging.info("data::: %s", data)
