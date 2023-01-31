#!usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import framework.utils
from framework.component import Component
import logging


class Pipeline:

    def __init__(self, conf_path):
        with open(conf_path) as fh:
            data = json.load(fh)
        self._components = []
        self._output_keys = data['output_keys']
        for component_conf in data['components']:
            self._add_component(component_conf)

    def _add_component(self, component_conf):
        name = component_conf["name"]
        conf = component_conf["conf"]
        cls_dict = framework.utils.get_subclass_dict(Component)
        cls = cls_dict[name]
        component = cls(conf)
        self._components.append(component)

    def restore(self):
        for component in self._components:
            component.restore()

    def process(self, message):
        for component in self._components:
            component.process(message)
        result = []
        if isinstance(message, list):
            for m in message:
                r_d = {}
                for key in self._output_keys:
                    r_d[key] = m.get(key)
                result.append(r_d)
        else:
            r_d = {}
            for key in self._output_keys:
                r_d[key] = message.get(key)
            result.append(r_d)
        return result

    def train(self, message_list):
        for component in self._components:
            component.train(message_list)


class PipelineFactory(object):

    _pipeline_dict = None

    def __init__(self):
        pass

    @classmethod
    def load(cls, conf_directory):
        if cls._pipeline_dict is not None:
            logging.warning("PIPELINE_FACTORY_ALREADY_LOADED_WARN")
        cls._pipeline_dict = {}
        logging.debug("LOAD_PIPELINE_FACTORY_CONF_DEBUG:path={}".format(conf_path))
        for file_name in os.listdir(conf_directory):
            if not file_name.endswith(".json"):
                continue
            pipeline = Pipeline(file_name)
            file_name = file_name.split("/")[-1][:-5]
            cls._pipeline_dict[file_name] = pipeline

    @classmethod
    def process(cls, pipeline_id, message):
        if pipeline_id not in cls._pipeline_dict:
            logging.error("WRONG_PIPELINE_ERROR:pipeline_id={}".format(pipeline_id))
            return
        pipeline = cls._pipeline_dict[pipeline_id]
        return pipeline.process(message)
