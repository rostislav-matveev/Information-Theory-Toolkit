#!/usr/bin/python3.11
import os
import gzip
import pickle
import sys

import logging
msg = logging.getLogger(__name__)
msg.setLevel(logging.DEBUG)

class _Tail(list):
    def __init__(self,maxlength = 10):
        pass
    
    def append(self,value):
        try:
            while True:
                self.remove(value)
        except ValueError:
            pass
        super().append(value)

    def truncate(self,max_length):
        ind = len(self) - max_length 
        if ind <= 0:
            return list()
        r = self[0:ind]
        del self[0:ind]
        return r
        

class persistent_dict(dict):
    _suffix = ".pickle.gz"
    
    def __init__(self,dirname,mincachesize=10,maxcachesize=None,ignorelock=False):
        if not os.path.isdir(dirname):
            if os.path.isfile(dirname):
                raise ValueError(f"Can not create directory {dirname}. "
                                 f"File exists and is not a directory. ")
            os.mkdir(dirname)
        self._dirname = dirname
        self._lockfile = os.path.join(dirname,"lock")
        if ( not ignorelock and
             os.path.isfile(self._lockfile) ):
            raise ValueError(f"Directory {dirname} is locked by another library. "
                             f"If not, inspect and remove file {dirname}/lock.")

        with open(self._lockfile,"w") as f:
             f.write(f"Library in {sys._getframe(1).f_locals['__name__']}\n")


             
        mincachesize = int(mincachesize)
        if mincachesize < 1: mincachesize = 1
        if maxcachesize == None or maxcachesize <= mincachesize + 1:
            maxcachesize = 2*mincachesize
        self.mincachesize = mincachesize
        self.maxcachesize = maxcachesize
        self._used = _Tail()

    def __del__(self):
        self.clear()
        try:
            os.remove(self._lockfile)
        except FileNotFoundError:
            pass

        
    @property
    def dirname(self):
        return self._dirname

    def _path(self,key):
        return os.path.join(self._dirname,key)+self._suffix
    
    def _normalize(self,key):
        if not isinstance(key,tuple):
            key = (key,)
        key_str = [str(k).strip() for k in key]
        return "_".join(key_str)

    def _truncate(self):
        if len(self)<=self.maxcachesize:
            return
        oldkeys = self._used.truncate(self.mincachesize)
        for k in oldkeys:
            del self[k]
        
    def __getitem__(self,key):
        key = self._normalize(key)
        try:
            result = super().__getitem__(key)
            self._used.append(key)
            return result
        except KeyError as ke:
            path = self._path(key)
            if not os.path.isfile(path):
                raise ke
            
            super().__setitem__( key, self._load(key) )
            self._used.append(key)
            self._truncate()
            return super().__getitem__(key)
                
    def __setitem__(self,key,value):
        key = self._normalize(key)
        super().__setitem__(key,value)
        self._used.append(key)
        self._truncate()
        self._dump(key,value)

    def _dump(self,key,value):
        path = self._path(key)
        with gzip.open(path,"wb") as f:
            pickle.dump(value,f)

    def _load(self,key):
        path = self._path(key)
        with gzip.open(path,"rb") as f:
            return pickle.load(f) 

    def keys(self):
        return [ f.replace(self._suffix,"")
                 for f in os.listdir(self._dirname)
                 if f.endswith(self._suffix) ]

    def cachedkeys(self):
        return super().keys()

    def items(self):
        for key in self.keys():
            yield (key,self.__getitem__(key))

    def values(self):
        for key in self.keys():
            yield self.__getitem__(key)

    def __len__(self):
        return len([ 0
                     for f in os.listdir(self._dirname)
                     if f.endswith(self._suffix) ])


        
class _TMP:
    def __getitem__(self,key):
        print(type(key))
        print(key)

    def __len__(self):
        return 100
    
