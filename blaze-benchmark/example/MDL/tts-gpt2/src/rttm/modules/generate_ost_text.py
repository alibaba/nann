#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/17
# update: 2020/9/17
"""
generate_ost_text.py:
"""

import json
from framework.component import Component
from rttm.utils.utils import call_ost_server


class GenerateOSTText(Component):
    """
    require: ["item_title"]
    provide: ["text"]
    """

    @classmethod
    def name(cls):
        return "generate_ost_text"

    def __init__(self, conf_dict):
        # super.__init__(conf_dict)
        pass

    def process(self, message):
        if isinstance(message, list):
            for m in message:
                title = m.get("item_title")
                max_length = int(m.get("max_length")) if m.get("max_length") else 8
                min_length = int(m.get("min_length")) if m.get("min_length") else 6
                count = int(m.get("count")) if m.get("count") else 5
                result = call_ost_server(max_length, min_length, count, "online_short_title", title, "")
                result = result.text.replace("cc(", "").replace(")", "")
                result = json.loads(result)["data"][0]["data"]
                text_list = []
                for d in  result:
                    text_list.append(d["title"])
                m.set("text", text_list)
        else:
            title = message.get("item_title")
            max_length = int(message.get("max_length")) if message.get("max_length") else 8
            min_length = int(message.get("min_length")) if message.get("min_length") else 6
            count = int(message.get("count")) if message.get("count") else 5
            result = call_ost_server(max_length, min_length, count, "online_short_title", title, "")
            result = result.text.replace("cc(", "").replace(")", "")
            result = json.loads(result)["data"][0]["data"]
            text_list = []
            for d in  result:
                text_list.append(d["title"])
            message.set("text", text_list)
