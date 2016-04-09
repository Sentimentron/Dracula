__author__ = 'rtownsend'

from nn_serialization import load_params
import types

def list_to_js(l):
    if type(l) != types.ListType:
        raise ValueError()
    if type(l[0]) == types.ListType:
        ret = []
        for i in l:
#            ret.append(list_to_js(i))
            inner = list_to_js(i)
            inner = 'Float32Array.from({})'.format(inner)
            ret.append(inner)
        return list_to_js(ret)
    l = ','.join([str(i) for i in arr])
    return '[{}]'.format(l)

def dict_to_js(name, d):
    print '{}'.format(name) + '= {'
    for c in d:
      a = c
      v = d[c]
      def escape(p):
        if '\\' in p:
            p = '\\\\'
        if '\'' in p:
            p = '\\\''
        return p
      if type(a) != type(0):
        a = "'" + escape(a) + "'"
      if type(v) != type(0):
        v = "'" + escape(v) + "'"
      print "\t{}: {},".format(a, v)
    print '};'

if __name__ == "__main__":
   params = load_params("lstm_model.npz", {})
   for p in params:
       if True:
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
            print 'draculaParams_{}={};'.format(p, arr)

   char_dict = params['char_dict']
   dict_to_js('draculaParams_char_dict', char_dict)

   inv_pos_dict = {}
   pos_dict = params['pos_dict']
   for pos in pos_dict:
       inv_pos_dict[pos_dict[pos]] = pos

   dict_to_js('draculaParams_inv_pos_dict', inv_pos_dict)

