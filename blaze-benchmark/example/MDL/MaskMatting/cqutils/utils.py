import time

def localtime():
    """local time: yyyy-mm-dd hh:mm:ss"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
