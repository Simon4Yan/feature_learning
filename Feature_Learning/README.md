# Vehicle ReID_baseline
Baseline model (with bottleneck) for vehicle ReID (using softmax and triplet loss).

## learning model
This part is for learning feature extractor. The code is modified form [reid_baseline](https://github.com/L1aoXingyu/reid_baseline), you can check each folder's purpose by yourself.

- **Train and test on VeRi dataset**
###### Train
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VeRi.yml' MODEL.NAME 'densenet121' `
###### Test
`CUDA_VISIBLE_DEVICES=1 python3 tools/test.py --config_file='configs/softmax_triplet_VeRi.yml' TEST.WEIGHT './CHECKPOINTS/VeRi/softmax_triplet/XX.pth' MODEL.NAME 'densenet121' `

- **Train and test on Aicity dataset**
The dataloader is in *./data/datasets/VR.py*. 
###### Train
During the training, the test dataset is unavailable, we adopt the test dataset of Veri as the validation. This means that training a model on Aicity and testing it on Veri.
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VR.yml' MODEL.NAME 'densenet121' `
###### Test
 Extract the feature representations of test images of Aicity dataset.
`CUDA_VISIBLE_DEVICES=0 python3 tools/update.py --config_file='configs/softmax_triplet_VR_test.yml' TEST.WEIGHT './CHECKPOINTS/VR/softmax_triplet/XX.pth' MODEL.NAME 'densenet121'`

## Feature expansion
This part consists of gallery expansion and query expansion.
###### gallery
`python3 gallery_feature.py `
###### Train
`python3 query_expansion.py `