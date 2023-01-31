# TFS client for python

import logging
import httplib2
import random
import hashlib
import datetime 
try :
    import json
except ImportError:
    import simplejson as json
import mmh

class TFSClient:
    """TFS Webservice Client Operator class."""
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger', logging.getLogger())
        self.app_key = kwargs.get('app_key', None)
        self.host = kwargs.get('root_server', None)
        self.rest = RestHelper(logger=self.logger)
        self.uid = kwargs.get('uid', None)
        if not self._initNgxServerList():
            raise Exception('init ngx server list failed.')
    def _initNgxServerList(self):
        """init ngx server list from root_server"""
        self.ngx_server_list = []
        self.ngx_req_limit = 0
        ret = self.rest.get(self.host, '/url.list', {})
        if ret == False:
            return False
        print('ret type before:{}'.format(type(ret)))
        if not isinstance(ret, str):
            ret = ret.decode('ascii')
        print('ret type after:{}'.format(type(ret)))
        print('ret:{}'.format(ret))
        _flist = ret.strip().split("\n")

        self.ngx_req_limit = _flist.pop(0)

        for l in _flist:
            if l.find(':') < 0:
                continue
            self.ngx_server_list.append(l)
        self._setNgxServer()
        print('_initNgxServerList done')
        return True
    def _setNgxServer(self):
        self.ngx_server = self.ngx_server_list[random.randint(0, len(self.ngx_server_list) - 1)]
    def _checkAppid( self ) :
        if not hasattr( self , "app_id" ) :
            self.logger.info('get appid from tfs')
            ret = self.rest.get( self.ngx_server , "/v2/{0}/appid".format( self.app_key ) , {} , [ 200 , 400 , 500 ] )
            ret = self.retHandler( ret );
            setattr( self , "app_id" , ret['APP_ID'] )
    def writeFile(self, content , suffix = ".jpg"):
        """write content to tfs, return is file name"""
        size = len(content)
        ret = self.rest.post(self.ngx_server, "{0}?suffix={1}".format( self.getUrl() , suffix ), content, {'Content-Length': str(size)})
        json_obj = self.retHandler( ret )
        file_name = json_obj['TFS_FILE_NAME']
        self.logger.info('Upload Tfs file:%s size:%s' % (file_name, size))
        return file_name
    def retHandler( self , ret ) :
        if ret == False:
            self.logger.error('write file to tfs failed.')
            raise Exception('write file to tfs failed.')
        try:
            json_obj = json.loads(ret)
        except Exception as e:
            self.logger.error('decode tfs return failed: %s' % e)
            raise Exception('decode tfs return failed: %s' % e)
        if not json_obj:
            self.logger.error('tfs return is empty')
            raise Exception('tfs return is empty')
        return json_obj
    @staticmethod
    def generateUIDByFilename( filename ) :
        mmhash = mmh.hash( filename ) % 1000000
        return mmhash + 1
    def _getUid( self , filename ) :
        if self.uid is None :
            return TFSClient.generateUIDByFilename( filename )
        else :
            return self.uid 
    def writeCustomFile( self , content , filename ) :
        self.rest.post( self.ngx_server , self.getUrl( filename ) , content , {'Content-Length': str( len( content ) ) },[201] )
        return "{0}/{1}/{2}".format( self.app_id , self._getUid( filename ) , filename )
    def updateCustomFile( self , content , filename ) :
        try :
            self.delCustomFile( filename )
        except Exception  :
            pass
        self.rest.put( self.ngx_server , "{0}?offset=0".format( self.getUrl( filename ) ) , content , {'Content-Length': str( len( content ) ) }  )
        return True
    def delCustomFile( self , filename ) :
        ret = self.rest.delete( self.ngx_server , self.getUrl( filename )  , {} )
        if ret == false :
             self.logger.error('delete file:%s from tfs failed.' % file_name)
             raise Exception('delete file:%s from tfs failed.' % file_name)
        return True
    def delFile(self, file_name):
        """delete file from tfs"""
        ret = self.rest.delete(self.ngx_server, self.getUrl() + '/' + file_name, {}, [200, 404])
        if ret == False:
            self.logger.error('delete file:%s from tfs failed.' % file_name)
            raise Exception('delete file:%s from tfs failed.' % file_name)
        self.logger.info('Delete Tfs file: %s' % file_name)
        return True

    def readFile(self, filename, suffix = "", offset = 0, size = None) :
        if size is None :
            metaMessage = self.getFileMeta(filename=filename, suffix=suffix)
            print('metaMessage:{}'.format(metaMessage))
            if metaMessage is None :
                return None
            else :
                metaMessage = self.retHandler(metaMessage) 
                size = metaMessage['SIZE']
        return self.rest.get( self.ngx_server , "/v1/{appkey}/{filename}?offset={offset}&size={size}&suffix={suffix}".format(
                    appkey = self.app_key ,
                    filename = filename ,
                    suffix = suffix ,
                    offset = offset ,
                    size = size ),
                    {} , [200] 
        )
    def readCustomFile( self , filename , offset = 0 , size =1024 , uid = None ) :
        if uid is None :
            ''' uid is not fixed '''
            uid = self._getUid( filename )
        self._checkAppid()
        return self.rest.get( self.ngx_server , "{url}?offset={offset}&size={size}".format( 
                    url = self.getUrl( filename ) ,
                    size = size ,
                    offset = offset
                ) ,
                {} ,
                [200]
        )
    def getFileMeta(self, filename, suffix="", force=0):
        if force != 0 :
            force = 1 
        print('getFileMeta: self.ngx_server:{}'.format(self.ngx_server))
        return self.rest.get(self.ngx_server,
                "/v1/{appkey}/metadata/{filename}?suffix={suffix}&force={force}".format(
            appkey=self.app_key, filename=filename, suffix=suffix, force=force),
                {}, [200])
    def getUrl(self , filename = None ):
        if filename is None :
            return '/v1/' + str(self.app_key)
        else :
            self._checkAppid()
            return '/v2/{appkey}/{appid}/{uid}/file/{filename}'.format( appkey = self.app_key , appid = self.app_id , uid= self._getUid( filename ) , filename = filename )

    def __del__(self):
        del self.logger
        del self.rest


class RestHelper:
    """http rest web service helper"""
    def __init__(self, **kwargs):
        """Restful webservice helper class."""
        self.logger = kwargs.get('logger', logging.getLogger())
        self.http = httplib2.Http(timeout=2)

    def get(self, host, url, headers, allow_code=[200]):
        return self._connect(host, url, '', 'GET', headers, allow_code)

    def post(self, host, url, body, headers, allow_code=[200]):
        return self._connect(host, url, body, 'POST', headers, allow_code)

    def put(self, host, url, body, headers, allow_code=[200]):
        return self._connect(host, url, body, 'PUT', headers, allow_code)

    def delete(self, host, url, headers, allow_code=[200]):
        return self._connect(host, url, '', 'DELETE', headers, allow_code)

    def _connect(self, host, url, body, method, headers, allow_code):
        """send http request to host"""
        _compact_url = 'http://' + host + '' + url
        print('_compact_url:{}'.format(_compact_url))
        # headers={'cache-control':'no-cache'}
        try:
            print("**********rest failed:: host: %s;; url: %s;; body: -;; method: %s;; headers: %s;;error: %s"% (host, url, method, headers, ''))
            (response, content) = self.http.request(_compact_url, method=method, body=body, headers=headers)
            print('response:{}'.format(response))
        except Exception as e:
            print('e:{}'.format(e))
            self.logger.error("rest failed:: host: %s;; url: %s;; body: -;; method: %s;; headers: %s;;error: %s"
                              % (host, url, method, headers, e))
            return False

        if not (response.status in allow_code):
            self.logger.error("rest reponse error:: host: %s;; url: %s;; body: -;; method: %s;; headers: %s;; reponse header: %s;; reason: %s;;allow: %s"
                              % (host, url, method, headers, response.status, response.reason, allow_code))
            return False

        self.logger.debug("rest request host: %s;; url: %s;; body: -;; method: %s;; headers: %s;; reponse header: %s;; reason: %s;;"
                          % (host, url, method, headers, response.status, response.reason))
        print('content type:{}'.format(type(content)))

        # if not isinstance(content, str):
        #     import chardet
        #     print(chardet.detect(content))
        #     # content = content.decode('gb2312')
        print('http request sucessed')
        return content

    def __del__(self):
        del self.logger
        del self.http



def test():
    import numpy as np
    import cv2
    tfs_client = TFSClient(app_key='52413f88bcefe',
        root_server='restful-store.vip.tbsite.net:3800', 
        logger=logging.getLogger())
    tfs = 'TB2R6x3bFuWBuNjSspnXXX1NVXa_!!687179516.jpg'
    img_content = tfs_client.readFile(tfs+'\n')
    img_data = np.asarray(bytearray(img_content), dtype='uint8')
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    print('read image size:{}'.format(img.shape))
                         

if __name__ == '__main__':
    print('Test for TFSClient start...')
    test()
