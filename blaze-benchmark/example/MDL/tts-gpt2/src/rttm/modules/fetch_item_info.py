#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/10
# update: 2020/9/10
"""
fetch_item_info.py:
"""

from framework.component import Component
from hsfpy import HsfContext, HsfConsumer, JavaObject, JavaWrapperObject, java2hsf
import logging
import time

class FetchItemInfo(Component):
    """
    require: ["item_id"]
    provide: ["item_title"]
    """

    @classmethod
    def name(cls):
        return "fetch_item_info"

    def __init__(self, conf_dict):
        # super.__init__(conf_dict)
        self.ctx = HsfContext('commonconfig.taobao.net', rpc_remoting_enable=True, hsf_response_timeout=15000, hsf_log_level="INFO", hsf_log_file="./hsflog.log")
        self.c = HsfConsumer('com.taobao.item.service.ItemQueryService', '1.0.0', 'HSF')

    def get_query_obj(self):
        query_opt = JavaObject('com.taobao.item.domain.query.QueryItemOptionsDO')
        query_opt.includeSkus = False
        query_opt.includeImages = False
        query_opt.autoMergeUser = False
        query_opt.includeItemExtends = False
        query_opt.includeQuantity = False
        query_opt.changePropertyIdToText = True
        return query_opt

    def process(self, message):
        try:
            query_opt = self.get_query_obj()

            if isinstance(message, list):
                for m in message:
                    # integer_object = JavaWrapperObject('java.lang.Integer', int(item_id))
                    ret = self.c.invoke('queryItemById', int(m.get("item_id")), query_opt)
                    hsf = java2hsf(ret)
                    title = hsf["item"]["title"]
                    pict_url = "https://img.alicdn.com/imgextra/" + hsf["item"]["pictUrl"]
                    prop_list = []
                    for prop in hsf["item"]["itemProperties"]:
                        prop_list.append((prop["propertyText"], prop["valueText"]))
                    m.set("item_title", title)
                    m.set("item_pict_url", pict_url)
                    m.set("item_prop_list", prop_list)
                    logging.info('### after fetch_item_info messages: %s', m._data)
            else:
                ret = self.c.invoke('queryItemById', int(message.get("item_id")), query_opt)
                hsf = java2hsf(ret)
                title = hsf["item"]["title"]
                pict_url = "https://img.alicdn.com/imgextra/" + hsf["item"]["pictUrl"]
                prop_list = []
                for prop in hsf["item"]["itemProperties"]:
                    prop_list.append((prop["propertyText"], prop["valueText"]))
                message.set("item_title", title)
                message.set("item_pict_url", pict_url)
                message.set("item_prop_list", prop_list)
                logging.info('### after fetch_item_info messages: %s', m._data)
        except Exception as e:
            logging.error(e)
            raise Exception("fetch item info failed! please check the item id")

        return message
