# 代码说明
## 录制数据集
```
python tool/datacreateskelenton.py 
```
+ skeleton格式
+ S开始录制
+ E停止录制
+ Q退出窗口
### 手动改保存路径
```
save_to_skeleton('../data/graph/1.skeleton', hand_data_list)
```
## 观看数据集
```
python tool/showskeleton.py 
```
### 手动改数据集路径
```
data_path = './data/1/hand_dataA011.skeleton'  
```
## 制作数据集
```
python tool/tools/utils/read_skeleton.py
```
+ input skeleton
+ output npy(数据)+pkl（标签）
## 检验数据集
```
python test.py
```
+ 可以读npy和pkl文件 自己改路径
## 训练
```
CUDA_VISIBLE_DEVICES=1 python main.py recognition -c config/st_gcn/hand_rgbd/train.yaml
```
+ 直接在服务器上面训练

### 训练参数文件
```
work_dir: ./work_dir/recognition/hand_rgbd/ST_GCN

# feeder
feeder: feeder.feeder.Feeder
train_feeder_args:  <-改数据集路径
  data_path: ./data/hand_rgbd/train_data.npy
  label_path: ./data/hand_rgbd/train_label.pkl
test_feeder_args:
  data_path: ./data/hand_rgbd/val_data.npy
  label_path: ./data/hand_rgbd/val_label.pkl

# model
model: net.st_gcn.Model
model_args:
  in_channels: 3
  num_class: 3 <-改类别数
  dropout: 0.5
  edge_importance_weighting: True
  graph_args:
    layout: 'ntu-rgb+d' <-改使用的图
    strategy: 'spatial'  <-改使用的连接方式

#optim
weight_decay: 0.0001
base_lr: 0.1
step: [10, 50]

# training
device: [0,1,2,3]
batch_size: 64 
test_batch_size: 64
num_epoch: 80
```
### 最后输出.pt文件

## 实时识别

```
python main.py demo
```
参数文件在
```
config/st-gcn/kinetics-skeleton/demo_realtime.yaml
```
标签文件在
```
resource/hand_test/label_name.txt
```
*不同模型对应的类别数目不一样 若num_class对不上则修改*
```

recognition yaml的
demo_realtime yaml的
label_list txt
self.fcn = nn.Conv2d(256, 5, kernel_size=1)
```