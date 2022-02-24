# Multi-Task Learning
This repo try to build MT model by opensource code from github

## Tasks
1. Semantic segmentation
2. Depth Estimation

## Datasets
1. Cityscapes

## Models
<details><summary> <b>Expand</b> </summary>

```
Laugh u
```

</details>

## Prepare
<details><summary> <b>Expand</b> </summary>

**1. Prepare Dataset from Cityscapes and merge depth to semantic**
- [Image](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
- [Semantic](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
- [Depth](https://www.cityscapes-dataset.com/file-handling/?packageID=7)

**2. Enviroment**
- python 3.6.9
- cuda 10.2
- torch 1.8.1
- 2 x 12G GPUs (e.g. GTX 1080ti)

</details>

## Train
```
python train.py --device 0,1 --name ${your name} --epochs 100 --input_height 384 --input_width 768 --batch-size 4 --random-flip --random-crop
```

## Val
```
python3 val.py --device 0,1 --weight ${your weight} --plot --input_height 1024 --input_width 2048 --batch-size 2
```

## Licenses
This is released under the MIT License (refer to the LICENSE file for details).

## Acknowledgements
[CCNet: Criss-Cross Attention for Semantic Segmentation](https://github.com/speedinghzl/CCNet)

[From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cleinc/bts)

[You Only Learn One Representation: Unified Network for Multiple Tasks](https://github.com/WongKinYiu/yolor)
