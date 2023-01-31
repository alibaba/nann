#!usr/bin/env python
# -*- coding: utf-8 -*-


class Component(object):

    def __init__(self, conf_dict):
        pass

    @classmethod
    def name(cls):
        raise NotImplementedError

    def restore(self):
        pass

    def process(self, message):
        raise NotImplementedError


class Message(object):

    def __init__(self, data={}):
        self._data = data

    def set(self, key, val):
        self._data[key] = val

    def get(self, key, default=None):
        return self._data.get(key, default)
