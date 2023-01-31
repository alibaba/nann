#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/17
# update: 2020/9/17
"""
quality_inspection.py:
"""

from framework.component import Component
from rttm.modules.qic.quality_inspection_control import quality_inspection_control


class GenerateText(Component):
    """
    require: ["origin_text"]
    provide: ["quality_text"]
    """

    @property
    def name(self):
        return "generate_text"

    def __init__(self, conf_dict):
        super.__init__(conf_dict)

    def process(self, message):
        origin_text = message.get("origin_text")
        quality_text = quality_inspection_control(origin_text)
        message.set("quality_text", quality_text)
