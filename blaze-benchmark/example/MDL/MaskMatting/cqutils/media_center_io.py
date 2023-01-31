#!/usr/bin/env python
#coding:utf8

from __future__ import print_function
# import sys
import os
import base64
import copy
import json
import configparser
import time
import requests
# from retrying import retry
import logging as logger

def retry_error(exception):
    retry_flag = isinstance(exception, requests.exceptions.Timeout)
    if retry_flag:
        logger.warning("retry cause:{}".format(str(exception)))
    return retry_flag

class MediaCenterIO():
    """
    put/read data via media center
    """
    def __init__(self, ownerid, bizcode, queryurl=None):
        logger.debug("initing picture center client")
        self.sess = requests.Session()
        self.headers = {
            "Http-Rpc-Type":"JsonContent",
            "Http-Rpc-Timeout":"6000",
            "Content-Type":"application/json"
        }
        # self.url = self.hostport + "/com.taobao.media.api.hsf.MediaService/1.0.0.daily/excute"
        self.defaulturl = 'http://mediacenter.vipserver:12220/com.taobao.media.api.hsf.MediaService/1.0.0/excute'
        if queryurl:
            self.url = queryurl
        else:
            self.url = self.defaulturl

        # 配置固定的参数，真正上传时只需要set fileData,name,size即可
        self.add_file_mode = \
        {
            "argsTypes": [
                "com.taobao.media.api.Request"
            ],
            "argsObjs": [
                {
                    "@type": "com.taobao.media.api.BaseRequest",
                    "command": "addFileAction",
                    "params": {
                        "skipNotify": False,
                        "clientVersion": {
                            "@type": "com.taobao.media.common.ClientVersion$Version",
                            "build": 6,
                            "major": 1,
                            "minor": 0
                        },
                        "ownerId": ownerid,
                        "bizCode": bizcode,
                        "fileAddTO": {
                            "@type": "com.taobao.media.to.FileAddTO",
                            "fileData": "",
                            "name": "",
                            "size": 0,
                            "userId": ownerid
                        }
                    }
                }
            ]
        }

    @classmethod
    def daily(cls):
        return cls(15702696, "tu",
                   "http://100.83.220.9:12220/com.taobao.media.api.hsf.MediaService/1.0.0.daily/excute")

    @classmethod
    def from_env(cls):
        ownerid = os.environ["MEDIA_CENTER_OWNER_ID"]
        bizcode = os.environ["MEDIA_CENTER_BIZCODE"]
        endpoint = os.environ.get("MEDIA_CENTER_ENDPOINT", None)
        return cls(ownerid, bizcode, endpoint)

    @classmethod
    def from_conf(cls, confpath=os.path.expanduser('~/.faiioutilconfig')):
        config = configparser.ConfigParser()
        config.read_file(open(confpath))
        ownerid = config.get("MediaCenter", "ownerID")
        bizcode = config.get('MediaCenter', 'bizCode')
        try:
            endpoint = config.get('MediaCenter', 'endpoint')
        except configparser.NoOptionError:
            endpoint = None
        return cls(ownerid, bizcode, endpoint)

    def get_post_body(self, tfs_data, target_name):
        body = copy.copy(self.add_file_mode)
        b64data = base64.b64encode(tfs_data).decode("utf-8")
        datalen = len(b64data)
        body["argsObjs"][0]["params"]["fileAddTO"]["fileData"] = b64data
        body["argsObjs"][0]["params"]["fileAddTO"]["size"] = datalen
        body["argsObjs"][0]["params"]["fileAddTO"]["name"] = target_name
        return json.dumps(body)

    # @retry(stop_max_attempt_number=3, wait_fixed=3000, stop_max_delay=6000, retry_on_exception=retry_error)
    def put_data(self, tfs_data, shorttfs=True):
        """
        put tfs_data to media center, and return the tfs cdn path.
        if shorttfs is True, return tfsname like 'O1CN014Fsw8N2GBW5lBnQYx_!!2477378977-0-faialgo.jpg'
        if shorttfs is False, return tfs url like 'i2/2477378977/O1CN014Fsw8N2GBW5lBnQYx_!!2477378977-0-faialgo.jpg'

        raise a 'requests.exceptions.Timeout' after try 3 times.
        """
        # pdb.set_trace()
        save_name = "faialgo_{}".format(int(time.time()*1000))
        body = self.get_post_body(tfs_data, save_name)
        resp = self.sess.post(self.url, data=body, headers=self.headers)
        if resp.status_code != 200:
            logger.error("failed to upload data, %s", resp.text)
            return None
        try:
            imgurl = resp.json()["data"]["result"]["url"]
        except json.decoder.JSONDecodeError:
            logger.error("failed to parse url from %s", resp.text)
            raise Exception(resp.text)
            return None
        except KeyError:
            logger.error("failed to parse url from %s", resp.text)
            raise Exception(resp.text)
            return None
        except:
            logger.error("failed to parse url from %s", resp.json()["data"])
            raise Exception(resp.json()["data"])
            return None

        if shorttfs:
            imgurl = imgurl.rsplit("/", 1)[-1]
        return imgurl

def main():
    # tpclient = MediaCenterIO.from_conf()
    tpclient = MediaCenterIO('0', 'scs')
    # tpclient = MediaCenterIO.daily()
    # logger.setLevel('DEBUG')
    # tfsdata = tfsclient.get_tfs_data('O1CN01I8K1CP2GBVy4TYHyP_!!2477378977-0-fai.jpg')
    imgdata = open("./test.jpg", "rb").read()
    try:
        imgurl = tpclient.put_data(imgdata)
    except Exception as e:
        print("Exception:{}".format(str(e)))
        return
    print(imgurl)
    # print(resp.content)

if __name__ == '__main__':
    main()
