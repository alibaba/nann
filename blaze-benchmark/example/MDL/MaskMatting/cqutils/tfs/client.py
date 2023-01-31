#coding: utf-8
__all__ = [
        'TfsClient',
        ]

import logging
import mmh3 as mmh
import codecs
import requests
import os.path as osp
try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json


def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


def loader(load_func=None, catch_exc=False, default=None):
    def _loader(f):
        def _fn(*args, **kwargs):
            lfunc = kwargs.pop('load_func') if 'load_func' in kwargs else load_func
            catchx = kwargs.pop('catch_exc') if 'catch_exc' in kwargs else catch_exc
            dflt = kwargs.pop('default') if 'default' in kwargs else default
            content = f(*args, **kwargs)
            if lfunc is None or not callable(lfunc):
                return content
            try:
                return lfunc(content)
            except Exception as e:
                if catchx:
                    return dflt
                raise e
        return _fn
    return _loader


class InitNgxServerListFailed(Exception):
    pass


class TfsMixin(object):

    def __init__(self, timeout=3):
        self.logger = logging.getLogger(__name__)
        self._base_url = 'http://{host}{interface}'
        self._timeout = timeout

    @loader(load_func=None)
    def do(self, method, host, interface, **kwargs):
        allow_code = kwargs.pop('allow_code', [200])
        headers = kwargs.pop('headers', {})
        timeout = kwargs.pop('timeout', self._timeout)
        url = self._base_url.format(host=host, interface=interface)
        func = getattr(requests, method.lower())
        resp = func(url, headers=headers, timeout=timeout, **kwargs)
        if not resp.status_code in allow_code:
            return None
        return resp.content

    def get(self, host, interface, **kwargs):
        return self.do('get', host, interface, **kwargs)

    def post(self, host, interface, **kwargs):
        return self.do('post', host, interface, **kwargs)

    def delete(self, host, interface, **kwargs):
        return self.do('delete', host, interface, **kwargs)

    def put(self, host, interface, **kwargs):
        return self.do('put', host, interface, **kwargs)

    @staticmethod
    def get_uid_by_filename(filename):
        return mmh.hash(filename) % 1000000 + 1


class TfsClient(TfsMixin):

    def __init__(self, app_key, root_server, logger=logging.getLogger(__name__), uid=None, timeout=4):
        super(TfsClient, self).__init__(timeout)
        self.logger = logger
        self.app_key = app_key
        self.host = root_server
        self.uid = uid
        self.__idx = 0

    def get_uid(self, filename):
        return self.get_uid_by_filename(filename) if self.uid is None else self.uid

    @lazy_property
    def ngx_serverlist(self):
        res = self.get(self.host, '/url.list')
        if res is None:
            raise InitNgxServerListFailed
        res = res.decode('ascii')
        flist = res.strip().split('\n')
        flist.pop(0)
        return [l for l in flist if l.find(':') >= 0]

    @property
    def ngx_server(self):
        ngx_list = self.ngx_serverlist
        try:
            cur_server = ngx_list[self.__idx]
            self.__idx = (self.__idx+1)%len(ngx_list)
            return cur_server
        except:
            ngx_list[0]

    @lazy_property
    def appid(self):
        d = self.get(self.ngx_server, "/v2/{0}/appid".format(self.app_key),
                allow_code=[200, 400, 500], load_func=json.loads)
        return d.get('APP_ID')

    def get_url(self, filename=None):
        if filename is None:
            return '/v1/{0}'.format(self.app_key)
        return "/v2/{appkey}/{appid}/{uid}/file/{filename}".format(
            appkey = self.app_key, appid = self.appid, uid = self.get_uid(filename), filename = filename
        )

    def read_file(self, filename, suffix="", offset=0, size=None, uid=None):
        if size is None:
            metainfo = self.get_file_meta(filename, suffix=suffix)
            if not metainfo:
                return None
            size = metainfo.get('SIZE')

        interface = "/v1/{appkey}/{filename}?offset={offset}&size={size}&suffix={suffix}".format(
            appkey = self.app_key, filename = filename, offset = offset, size = size, suffix = suffix
        )
        return self.get(self.ngx_server, interface)

    def save_file(self, filename, savepath, bulk=0):
        if bulk <= 0:
            with codecs.open(savepath, 'wb') as f:
                data = self.read_file(filename)
                data and f.write(data)
            return savepath

        offset = 0
        with codecs.open(savepath, 'wb') as f:
            while 1:
                bulk_data = self.read_file(filename, offset=offset, size=bulk)
                if bulk_data is None:
                    break
                f.write(bulk_data)
                offset += bulk
        return savepath

    def get_file_meta(self, filename, suffix="", force=0):
        force = 1 if force != 0 else force
        interface = "/v1/{appkey}/metadata/{filename}?suffix={suffix}&force={force}".format(
            appkey = self.app_key, filename = filename, suffix = suffix, force = force
        )
        return self.get(self.ngx_server, interface, catch_exc=True, load_func=json.loads, default={})

    def write_file(self, content, filename=None, suffix=".jpg"):
        size = len(content)
        # size = osp.getsize(filepath)
        # if size == 0:
        #     self.logger.warn('{0} size is 0'.format(filepath))
        #     return False

        # with codecs.open(filepath, 'rb') as f:
        #     content = f.read()

        headers = {'Content-Length': str(size)}
        if filename is None:
            interface = "{url}?suffix={suffix}".format(url=self.get_url(), suffix=suffix)
            d = self.post(self.ngx_server, interface, data=content,
                    headers=headers, load_func=json.loads)
            return d.get('TFS_FILE_NAME')

        self.post(self.ngx_server, self.get_url(filename), data=content, headers=headers, allow_code=[201])
        return '{appid}/{uid}/{filename}'.format(
            appid = self.appid, uid = self.get_uid(filename), filename = filename
        )

    def del_file(self, filename, custom=False):
        if not custom:
            interface = "{url}/{filename}".format(url=self.get_url(), filename=filename)
            content = self.delete(self.ngx_server, interface, allow_code=[200, 404])
            return content is not None

        content = self.delete(self.ngx_server, self.get_url(filename))
        return content is not None



if __name__ == '__main__':
    import cv2
    import numpy as np
    root_server = 'restful-store.vip.tbsite.net:3800'
    appkey = '52413f88bcefe'

    tfs = TfsClient(appkey, root_server, timeout=10)
    name = 'TB2R6x3bFuWBuNjSspnXXX1NVXa_!!687179516.jpg' 
    print (name)
    # meta = tfs.get_file_meta(name)
    # print('meta:{}'.format(meta))
    img_content = tfs.read_file(name)
    img_data = np.asarray(bytearray(img_content), dtype='uint8')
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    print('read image size:{}'.format(img.shape))
    suffix = '.jpg'
    img_content1 = cv2.imencode(suffix, img)[1]
    img_data1 = np.array(img_content).tostring()
    wname = tfs.write_file(img_data1)
    print(wname)
    # print (tfs.save_file(name, './test', bulk=1))
