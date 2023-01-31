#!usr/bin/env python
# -*- coding: utf-8 -*-


def get_all_subclasses(cls):
    """
    Returns all known (imported) subclasses of a class.
    """
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in get_all_subclasses(s)]


def get_subclass_dict(cls):
    """
    :param cls:
    :return: dict
    """
    all_subclasses = get_all_subclasses(cls)
    return {cls.name(): cls for cls in all_subclasses}
