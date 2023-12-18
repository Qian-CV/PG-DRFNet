import pickle
import numpy as np
path = '/media/ubuntu/CE425F4D425F3983/datasets/VEDAI_1024/trainval/trainval_noOther_s2anet.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)
a = np.array(data)
# print(data)
print(type(a))
# print(a[1, :])
print(data)
