# sasa
 
 This project provides the source code for the following two papers.
 
 [Using Statistics to Automate Stochastic Optimization](http://papers.nips.cc/paper/9150-using-statistics-to-automate-stochastic-optimization) (SASA)
 
 [Statistical Adaptive Stochastic Optimization](https://openreview.net/forum?id=B1gkpR4FDB) (SASA+ and SSLS)
 
 By providing several convenient tools (supporting various public datasets, supporting tensorboard,
 checkpointing, etc), we also hope that this repo can help researchers 
 quickly test their ideas of optimization for deep learning. 
 
 ## Prepare Dataset:
 This repo currently supports CIFAR10, CIFAR100, ImageNet, MNIST and Wikitext-2 datasets.
 In the default setting, the data should be stored in src/datasets/CIFAR10, src/datasets/CIFAR100, 
 src/datasets/imagenet, src/datasets/MNIST and src/datasets/wikitext-2, respectively.
 
 The CIFAR10, CIFAR100 and MNIST can be automatically downloaded.
 
 One needs to download and zip the train and validation images of ImageNet under the specified data folder, e.g., 
 the default src/datasets/imagenet. One also needs to download the Wikitext-2 dataset from, e.g., 
 https://github.com/pytorch/examples/tree/master/word_language_model/data/wikitext-2.
 
 With the default setting, we should have the following files in the src/datasets directory:
 ```
src (root folder)
├── datasets (folder with all the datasets and pretrained models)
├──── imagenet/ (imagenet dataset and pretrained models)
├────── 2012/
├───────── train.zip
├───────── val.zip
├───────── train_map.txt
├───────── val_map.txt
├──── wikitext-2/ (wikitext-2 dataset and pretrained models)
├────── train.txt
├────── valid.txt
├────── text.txt
├──── CIFAR10/ (CIFAR10 dataset and pretrained models)
├──── CIFAR100/ (CIFAR100 dataset and pretrained models)
├──── MNIST/ (MNIST dataset and pretrained models)
```
 
## Environment requirements:
It is recommended to use any of the following docker images to run the experiments.
```
pengchuanzhang/maskrcnn:py37cuda90
igitman/maskrcnn:py3.7-cuda-10-pytorch-1.2-openmpi
```
For virtual environments, the following packages should be the sufficient.
```
pytorch >= 0.4.1
tensorboardx tensorboard tensorflow (for tensorboard support)
```

## Training/Testing on Local Machine:
Navigate to the `src` folder, run the following without modification.

### CIFAR10
```
python run_experiment.py --config-file 'config/cifar10.yaml' --data ../datasets/CIFAR10 --output_dir ../run/CIFAR10/qhmtest OPTIM.OPT 'qhm'
python run_experiment.py --config-file 'config/cifar10.yaml' --data ../datasets/CIFAR10 --output_dir ../run/CIFAR10/yaidatest OPTIM.OPT 'yaida_ratio'
python run_experiment.py --config-file 'config/cifar10.yaml' --data ../datasets/CIFAR10 --output_dir ../run/CIFAR10/sasatest OPTIM.OPT 'yaida_seq' OPTIM.DROP_FACTOR 10.0 OPTIM.SASA.SIGMA 0.2 OPTIM.SASA.DELTA 0.02 OPTIM.SASA.TESTS_PER_EPOCH 1
python run_experiment.py --config-file 'config/cifar10.yaml' --data ../datasets/CIFAR10 --output_dir ../run/CIFAR10/sasaplustest OPTIM.OPT 'sasa_xd' OPTIM.DROP_FACTOR 10.0 OPTIM.SASA.SIGMA 0.01 OPTIM.SASA.LEAKY_RATIO 4 OPTIM.SASA.TESTS_PER_EPOCH 1
python run_experiment.py --config-file 'config/cifar10.yaml' --data ../datasets/CIFAR10 --output_dir ../run/CIFAR10/sgdslstest OPTIM.OPT 'sgd_sls' OPTIM.LS.GAMMA 0.05 OPTIM.LS.SDC 0.1 OPTIM.LS.IGN 1 OPTIM.LS.EVAL 1
```

### ImageNet
```
python run_experiment.py --config-file 'config/imagenet.yaml' --data ../datasets/imagenet/2012/ --output_dir ../run/imagenet/qhmtest OPTIM.OPT 'qhm'
python run_experiment.py --config-file 'config/imagenet.yaml' --data ../datasets/imagenet/2012/ --output_dir ../run/imagenet/yaidatest OPTIM.OPT 'yaida_ratio'
python run_experiment.py --config-file 'config/imagenet.yaml' --data ../datasets/imagenet/2012/ --output_dir ../run/imagenet/sasatest OPTIM.OPT 'yaida_seq' OPTIM.DROP_FACTOR 10.0 OPTIM.SASA.SIGMA 0.2 OPTIM.SASA.DELTA 0.02 OPTIM.SASA.TESTS_PER_EPOCH 1
python run_experiment.py --config-file 'config/imagenet.yaml' --data ../datasets/imagenet/2012/ --output_dir ../run/imagenet/sasaplustest OPTIM.OPT 'sasa_xd' OPTIM.DROP_FACTOR 10.0 OPTIM.SASA.SIGMA 0.01 OPTIM.SASA.LEAKY_RATIO 4 OPTIM.SASA.TESTS_PER_EPOCH 1
python run_experiment.py --config-file 'config/imagenet.yaml' --data ../datasets/imagenet/2012/ --output_dir ../run/imagenet/sgdslstest OPTIM.OPT 'sgd_sls' OPTIM.LS.GAMMA 0.05 OPTIM.LS.SDC 0.1 OPTIM.LS.IGN 1 OPTIM.LS.EVAL 1
```

### Wikitext-2
```
python run_experiment.py --config-file 'config/rnn.yaml' --data ../datasets/wikitext-2/ --output_dir ../run/wikitext/rnn/qhmtest OPTIM.OPT 'qhm' OPTIM.LR 20.0 OPTIM.MOM 0.0 OPTIM.NU 1.0 MODEL.RNN.CLIP 0.25 MODEL.RNN.INIT0 0 MODEL.RNN.SHUFFLE 0 DATALOADER.RE 'no'
python run_experiment.py --config-file 'config/rnn.yaml' --data ../datasets/wikitext-2/ --output_dir ../run/wikitext/rnn/adamtest OPTIM.OPT 'adam' OPTIM.LR 0.001 MODEL.RNN.CLIP 0.25 MODEL.RNN.INIT0 0 MODEL.RNN.SHUFFLE 0 DATALOADER.RE 'no' 
python run_experiment.py --config-file 'config/rnn.yaml' --data ../datasets/wikitext-2/ --output_dir ../run/wikitext/rnn/sasaplustest OPTIM.OPT 'sasa_xd' MODEL.RNN.INIT0 1 MODEL.RNN.SHUFFLE 1 DATALOADER.RE 'yes' OPTIM.SASA.SIGMA 0.01 OPTIM.SASA.LEAKY_RATIO 4 OPTIM.SASA.TESTS_PER_EPOCH 1
python run_experiment.py --config-file 'config/rnn.yaml' --data ../datasets/wikitext-2/ --output_dir ../run/wikitext/rnn/sgdslstest OPTIM.OPT 'sgd_sls' MODEL.RNN.INIT0 0 MODEL.RNN.SHUFFLE 0 DATALOADER.RE 'no' OPTIM.LS.GAMMA 0.05 OPTIM.LS.SDC 0.1
```

## Acknowledgement
1. The rnn `src/rnndata.py` and `src/rnnmodel.py` are modified from https://github.com/pytorch/examples/tree/master/word_language_model/.
2. The `src/mymodels` is modified from https://github.com/kuangliu/pytorch-cifar/tree/master/models.
