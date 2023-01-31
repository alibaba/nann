#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/17
# update: 2020/9/17
"""
risk_control.py:
"""

from framework.component import Component
from hsfpy import *


class RiskControl(Component):
    """
    require: ["text"]
    provide: ["text"]
    """

    @classmethod
    def name(cls):
        return "risk_control"

    def __init__(self, conf_dict):
        self.ctx = HsfContext('commonconfig.taobao.net', rpc_remoting_enable=True, hsf_response_timeout=15000, hsf_log_level="INFO", hsf_log_file="./hsflog.log")
        self.c = HsfConsumer('com.alimama.rcp.shield.api.sync.sdk.ShieldSyncRuleService', '1.0.0', 'HSF')

    def get_query_pbj(self):
        request = JavaObject('com.alimama.rcp.shield.api.sync.dto.SyncTextDTO')
        request.tokenId = JavaWrapperObject('java.lang.Long', 1179)
        return request

    def process(self, message):
        if isinstance(message, list):
            for m in message:
                risk_code_list = []
                text_list = m.get("text")
                request_list = []        
                for text in text_list:
                    query_obj = self.get_query_pbj()
                    query_obj.text = text
                    request_list.append(query_obj)
                ret = self.c.invoke('textAccessSyncRuleBatchJudge', request_list)
                ret = java2hsf(ret)
                for r in ret["obj"]:
                    risk_code_list.append(r["actionCode"])
                m.set("risk_code", risk_code_list)
        else:
            risk_code_list = []
            text_list = message.get("text")
            request_list = []        
            for text in text_list:
                query_obj = self.get_query_pbj()
                query_obj.text = text
                request_list.append(query_obj)
            ret = self.c.invoke('textAccessSyncRuleBatchJudge', request_list)
            ret = java2hsf(ret)
            for r in ret["obj"]:
                risk_code_list.append(r["actionCode"])
            message.set("risk_code", risk_code_list)
