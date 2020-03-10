'''
@Author: xujiahuan
@Date: 2020-03-10 09:16:08
@LastEditors: xujiahuan
@LastEditTime: 2020-03-10 17:02:01
'''

import pickle


def save_obj(obj, name):
    with open('obj/'+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


d = {"a": 2, "b": 3}
name = "d"
# save_obj(d, name)
dd = load_obj(name)
print(dd)
