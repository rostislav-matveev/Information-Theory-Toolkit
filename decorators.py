def function_decorator(d):
    save_attr = ( "__name__","__qualname__", "__doc__", "__annotations__" )
    def dsaveattr(fcn):
        if not isinstance(fcn,type(dsaveattr)):
            raise TypeError("{} decorates only functions".format(d.__name__))
        dfcn = d(fcn)
        for attr in save_attr:
            dfcn.__setattr__(attr, fcn.__getattribute__(attr))
        dfcn.__dict__.update(fcn.__dict__)
        return dfcn
    return dsaveattr

function_decorator=function_decorator(function_decorator)

@function_decorator
def class_decorator(d):
    save_attr = ( "__name__","__qualname__", "__doc__")
    def dsaveattr(c):
        if not isinstance(c,type):
            raise TypeError("{} decorates only classes".format(d.__name__))
        dc=d(c)
        for attr in save_attr:
            try:
                a=getattr(c,attr,1)
            except:
                continue
            setattr(dc,attr, a)
        return dc
    return dsaveattr


################################################################################
class fcn_cache(dict):
    def __init__(self,*,name=""):
        self.name   = name
        super().__init__()

    @property
    def hits(self):
        return sum(( x[1] for x in self.values() ))

    @property
    def misses(self):
        return len(self)

    def __str__(self):
        lines = list()
        lines.append("Cache for function {}".format(self.name))
        lines.append( "Hits: {}, Misses: {}".
                      format(self.hits,self.misses) )
        for a, ( v, n ) in self.items():
            if isinstance(v,Exception): v=repr(v)
            lines.append("{:3} x {}({}) -> {}".format(n,self.name,a,v))
        return "\n".join(lines)

@function_decorator
def cached(fcn):
    def cached_fcn(*args,**kwargs):
        call_signature=",".join([repr(a) for a in args] +
                                [repr(kwa[0])+"="+repr(kwa[1])
                                 for kwa in sorted(kwargs.items()) ])
        if call_signature not in cached_fcn.cache:
            try:
                val = fcn(*args,**kwargs)
            except Exception as err:
                val = err
                raise err
            finally:
                cached_fcn.cache[call_signature] = [val,0]
        else:
            cached_fcn.cache[call_signature][1] += 1
        if isinstance(cached_fcn.cache[call_signature][0],Exception):
            raise cached_fcn.cache[call_signature][0]
        else:
            return cached_fcn.cache[call_signature][0]
        
    cached_fcn.cache = fcn_cache(name=fcn.__name__)
    return cached_fcn


@class_decorator
def singleton(c:type) -> type:
    class dc(c):
        _singleton_instance = None
        def __new__(cls,*args,**kwargs):
            if cls._singleton_instance == None:
               cls._singleton_instance = super().__new__(dc,*args,**kwargs)
            return cls._singleton_instance
    return dc


################################################################################
@singleton
class A: pass

@cached
def fibfast(n):
    if n in (0,1): return 1
    if n==15: raise Exception
    
    return fibfast(n-1) + fibfast(n-2)



