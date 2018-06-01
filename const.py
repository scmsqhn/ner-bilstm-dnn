import sys
import os

"""
envs = dict()
with open("envs.json", "r") as f:
    envs = json.loads(f.read())
"""


class _const(object):
    class ConstError(PermissionError): pass
    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            raise self.ConstError( "Can't rebind const(%s)" % name)
        self.__dict__[name] = value


    def __delattr__(self, name):
        if name in self.__dict__:
            raise self.ConstError("Can't unbind const(%s)" % name)
        raise NameError(name)
        self.str2var()

    def lc(self, name, value):
        locals()[name] = value

    def str2var(self):
        for key in self.__dict__.keys():
            locals()[key] = self.__dict__[key]

#sys.modules[__name__] = _const()
#print(sys.modules[__name__])
const = _const()
const.__setattr__("SUCC", "success")
const.__setattr__("FAIL", "fail")
const.str2var()
print(const.SUCC)
print(const.FAIL)



