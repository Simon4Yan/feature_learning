# Vehicle ReID_baseline
Baseline model (with bottleneck) for vehicle ReID (using softmax and triplet loss).

## learning model
This part is for learning feature extractor. The code is modified form [reid_baseline](https://github.com/L1aoXingyu/reid_baseline), you can check each folder's purpose by yourself.

**Train**

CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_DE.yml' MODEL.NAME 'densenet121' 

**Test**

CUDA_VISIBLE_DEVICES=1 python3 tools/update.py --config_file='configs/softmax_triplet_test.yml' TEST.WEIGHT './CHECKPOINTS/VR/softmax_triplet/XX.pth' MODEL.NAME 'densenet121' 

## Feature expansion
This part consists of gallery expansion and query expansion.

## If you have any question, please contact us by E-mail (dengwj16@gmail.com) or open an issue in this project. Thanks.
