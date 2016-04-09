__author__ = 'rtownsend'

from nn_serialization import load_params
import types

def list_to_js(l):
    if type(l) != types.ListType:
        raise ValueError()
    if type(l[0]) == types.ListType:
        ret = []
        for i in l:
            ret.append(list_to_js(i))
        return list_to_js(ret)
    l = ','.join([str(i) for i in arr])
    return '[{}]'.format(l)

if __name__ == "__main__":
   params = load_params("lstm_model.npz", {})
   for p in params:
       if '_dict' in p:
           continue
       arr = params[p]
       if type(arr) == type(0):
           continue
       if type(arr) == type(0.):
           continue
       arr = arr.tolist()
       if type(arr) == type(0):
           continue
       if type(arr) == type(0.):
           continue
       arr = list_to_js(arr)
       print '{}={};'.format(p, arr)

   char_dict = params['char_dict']
   print 'char_dict = {'
   for c in char_dict:
      a = c
      if '\\' in c:
          a = '\\\\'
      if '\'' in c:
          a = '\\\''
      print "\t'{}': {},".format(a, char_dict[c])
   print '};'
