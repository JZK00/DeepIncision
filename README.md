# DeepIncision
VLFM-Driven Efficient Recognition of Surgical Incisions


***Environment***
This repo requires Pytorch>=1.9 and torchvision. We recommand using docker to setup the environment. You can use this pre-built docker image ``docker pull pengchuanzhang/maskrcnn:ubuntu18-py3.7-cuda10.2-pytorch1.9`` or this one ``docker pull pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`` depending on your GPU.

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers 
python setup.py build develop --user
```

***Backbone Checkpoints.*** Download the ImageNet pre-trained backbone checkpoints into the ``MODEL`` folder. 
```
mkdir MODEL
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/swin_tiny_patch4_window7_224.pth -O swin_tiny_patch4_window7_224.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth
```

***Command.***
train see
```
bash surgin_train.sh
```


val see
```
bash test.sh
```

metric
```
python metric.py
```


emsemble see
```
python metric_em.py
```



based on: GLIP: Grounded Language-Image Pre-training  
