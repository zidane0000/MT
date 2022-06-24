# Multi-Task Learning
This repo try to build MT model by opensource code from github

## Tasks
1. Semantic segmentation
2. Depth Estimation
3. Object detection

## Datasets
1. Cityscapes

## Models
<details><summary> <b>Expand</b> </summary>

![proposed model](https://github.com/zidane0000/MT/blob/main/figure/proposed%20model%20v2.jpg)

</details>

## Prepare
<details><summary> <b>Expand</b> </summary>

**1. Prepare Dataset from Cityscapes and merge depth to semantic**
- [Image](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
- [Semantic](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
- [Depth](https://www.cityscapes-dataset.com/file-handling/?packageID=7)
- Object Detection(Unaviable)

**2. Enviroment**
- python 3.6.9
- cuda 10.2
- torch 1.8.1

</details>

## Train
### Mutli-Task
```
python train.py --device 0,1,2,3 \
    --root {your dataset path} \
    --random-flip --random-crop --multi-scale \ # Optional
    --epochs 100 --save-cycle 90 \
    --input_height 512  --input_width 1024 \
    --encoder resnet50 \
    --semantic_head hrnet  --smnt_num_class 19 \
    --depth_head bts \
    --obj_head yolo  --obj_num_class 10 \
    --batch-size 16 \
    --warmup 3 \
    --adam \
    --name {your name}
```
### Single-Task (ex: object detection)
```
python train.py --device 0,1,2,3 \
    --root {your dataset path} \
    --random-flip --random-crop --multi-scale \ # Optional
    --epochs 100 --save-cycle 90 \
    --input_height 512  --input_width 1024 \
    --encoder resnet50 \
    --semantic_head ''  --smnt_num_class 19 \
    --depth_head '' \
    --obj_head yolo  --obj_num_class 10 \
    --batch-size 16 \
    --warmup 3 \
    --adam \
    --name {your name}
```

## Val
```
python3 val.py --device 0,1 --weight ${your weight} \
    --input_height 1024 --input_width 2048 \
    --batch-size 2 \
    --encoder resnet50 \ # Same as Train
    --semantic_head ''  --smnt_num_class 19 \ # Same as Train
    --depth_head '' \ # Same as Train
    --obj_head yolo  --obj_num_class 10 \ # Same as Train
    --plot # if want plot image
```

## Licenses
This is released under the MIT License (refer to the LICENSE file for details).

## Acknowledgements
[CCNet: Criss-Cross Attention for Semantic Segmentation](https://github.com/speedinghzl/CCNet)

[High-resolution networks and Segmentation Transformer for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)

[From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cleinc/bts)

[You Only Learn One Representation: Unified Network for Multiple Tasks](https://github.com/WongKinYiu/yolor)
