# Vehicle ReID_baseline
Baseline model (with bottleneck) for vehicle ReID (using softmax and triplet loss).

## learning model
This part is for learning feature extractor. The code is modified form [reid_baseline](https://github.com/L1aoXingyu/reid_baseline), you can check each folder's purpose by yourself.

- **TRAIN AND TEST ON VERI DATASET**
#####Train
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VeRi.yml' MODEL.NAME 'densenet121' `
#####Test
`CUDA_VISIBLE_DEVICES=1 python3 tools/test.py --config_file='configs/softmax_triplet_VeRi.yml' TEST.WEIGHT './CHECKPOINTS/VeRi/softmax_triplet/XX.pth' MODEL.NAME 'densenet121' `

- **TRAIN AND TEST ON AICITY DATASET**
#####Train
The dataloader is in *./data/datasets/VR.py*.
During the training, the test dataset is unavailable, we adopt the test dataset of Veri as the validation. This means that training a model on Aicity and testing it on Veri.
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VR.yml' MODEL.NAME 'densenet121' `
##### Test
 Extract the feature representations of test images of Aicity dataset.
`CUDA_VISIBLE_DEVICES=0 python3 tools/update.py --config_file='configs/softmax_triplet_VR_test.yml' TEST.WEIGHT './CHECKPOINTS/VR/softmax_triplet/XX.pth' MODEL.NAME 'densenet121'`
- **MODEL_ENSEMBLE**
To further improve the discriminative ability of features, we adopt model ensemble method.
###### 1. Model1 
Train model with default settings.
*train:*
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VR.yml' MODEL.NAME 'densenet121'  OUTPUT_DIR: "./CHECKPOINTS/VR/model1"`
*test:*
`CUDA_VISIBLE_DEVICES=0 python3 tools/update.py --config_file='configs/softmax_triplet_VR_test.yml' TEST.WEIGHT './CHECKPOINTS/VR/model1/XX.pth' MODEL.NAME 'densenet121' TEST.QF_NAME 'qf_1' TEST.GF_NAME 'gf_1'`
###### 2. Model2
Train model with extra data augmentation (COLORJITTER).
*train:*
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VR.yml' MODEL.NAME 'densenet121'  OUTPUT_DIR: "./CHECKPOINTS/VR/model2" INPUT.COLORJITTER 'True'`
*test:*
`CUDA_VISIBLE_DEVICES=0 python3 tools/update.py --config_file='configs/softmax_triplet_VR_test.yml' TEST.WEIGHT './CHECKPOINTS/VR/model1/XX.pth' MODEL.NAME 'densenet121' INPUT.COLORJITTER 'True'  TEST.QF_NAME 'qf_2' TEST.GF_NAME 'gf_2'`
###### 3. Model3
Use softmargin for the triplet loss.
*train:*
`CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config_file='configs/softmax_triplet_VR.yml' MODEL.NAME 'densenet121'  OUTPUT_DIR: "./CHECKPOINTS/VR/model3" DATALOADER.SOFT_MARGIN 'False'`
*test:*
`CUDA_VISIBLE_DEVICES=0 python3 tools/update.py --config_file='configs/softmax_triplet_VR_test.yml' TEST.WEIGHT './CHECKPOINTS/VR/model3/XX.pth' MODEL.NAME 'densenet121' TEST.QF_NAME 'qf_3' TEST.GF_NAME 'gf_3'`
###### 4. Concatenate
The features of three models are saved in './feature_expansion/data/'.
`python feature_concat.py`
## Feature expansion
This part consists of gallery expansion and query expansion.
#####gallery
`python3 gallery_feature.py `
##### Train
`python3 query_expansion.py `
