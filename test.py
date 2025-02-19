import numpy as np
data=np.load("/home/yuyu/workspace/sci/project/data/data_test/val_data.npy")
print(data.shape)
print(data.dtype)
# print(data)

if np.isnan(data).any():
    print("数据中存在NaN值！")
else:
    print("数据中不存在NaN值。")
import pickle
with open("/home/yuyu/workspace/sci/project/data/data_test/train_label.pkl", "rb") as f:
    label = pickle.load(f)
    print(label)
    print(type(label))
