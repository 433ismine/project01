import numpy as np
data=np.load("/home/yuyu/workspace/sci/ws/src/project01/data_test/data_all/train_record.json.npy", allow_pickle=True)
# print(data.shape)
# print(data.dtype)
print(data)

# if np.isnan(data).any():
#     print("数据中存在NaN值！")
# else:
#     print("数据中不存在NaN值。")
# import pickle
# with open("/home/yuyu/workspace/sci/project/data/data_test/train_label.pkl", "rb") as f:
#     label = pickle.load(f)
#     print(label)
#     print(type(label))


neighbor_1base = [(0, 1), (1, 2), (2, 3), (3, 4),
                  (0, 5), (5, 6), (6, 7), (7, 8),
                  (0,9),(9, 10), (10, 11), (11, 12),
                   (0,13),(13, 14), (14, 15), (15, 16),
                  (0, 17), (17, 18), (18, 19), (19, 20),
                  (0,20)
                  ]
# neighbor_1base = [(1, 2), (2, 3), (3, 4), (4, 5),
#                   (1, 6), (6, 7), (7, 8), (8, 9),
#                   (1, 10), (10, 11), (11, 12), (12, 13),
#                   (1, 14), (14, 15), (15, 16), (16, 17),
#                   (1, 18), (18, 19), (19, 20), (20, 21),
#                   (1, 21)
#                   ]