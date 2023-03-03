# Visually-Aware Audio Captioning Transformer (V-ACT)
This repository contains source code for our paper [Visually-Aware Audio Captioning with Adapeive Audio-Visual Attention](https://arxiv.org/pdf/2210.16428.pdf).
## Set up environment
* Create a conda environment with dependencies: `conda env create -f environment.yml -n name`
* All of our experiments are running on RTX 30 series GPUs with CUDA11. This environment may just work for RTX 30x GPUs.
## Set up dataset
All the experiments were carried out on AudioCaps dataset, which is sourced from AudioSet.
Our download version （video available) contains 47745/49837 audio clips in training set, 480/495 audio clips in validation set, 928/975 audio clips in test set.

For reproducibility, our downloaded version can be accessed at: 
* [Baidu](https://pan.baidu.com/s/1DkGsfQ0aM6lx6Gf6gCyrVw) password: a1p4 
* [Google Drive](https://drive.google.com/drive/folders/1e5v-u7qRtmKAzQVMbBSy7CU-tp1nA0su?usp=sharing)

To prepare the dataset:
* Put downloaded zip files under `data` directory, and run `data_unzip.sh` to extract the zip files.
* Run `python data_prep.py` to create h5py files of the dataset.

## Prepare evaluation tool

* Run `coco_caption/get_stanford_models.sh` to download the libraries necessary for evaluating the metrics.

## Experiments 

### Training

* The default setting is for 'ACT_m_scratch'
* Run experiments: `python train.py -m audio -n exp_name`
* Set the parameters you want in `settings/settings.yaml`
* Modality [only audio available now]: 
  * audio 
  * video 
  * audio-visual 
#### Pretrained encoder

We provide two pretrained encoders, one is a pretrained DeiT model, another is the DeiT model pretrained on AudioSet.
1. [DeiT model](https://drive.google.com/file/d/1eA3SYO2n9soU5AB6YxMNGDjno5E5AN7Q/view?usp=sharing)
2. [DeiT model pretrained on AudioSet](https://drive.google.com/file/d/1QgQLbeBHwly5UN_V15mSJZ812h6QIgFe/view?usp=sharing)

To use pretrained encoder:
* Download the pretrained encoder model and put it under the directory `pretrained_models`
* Set settings in `settings/settings,yaml`
  * set `encoder.model:` to 'audioset'
  * set `encoder.pretrained` to 'Yes'
  * set `path.encoder` to the model path, e.g. 'pretrained_models/audioset_deit.pth'
* Run experiments

## Cite
If you wish to cite this work, please kindly cite the following paper:
```
@article{liu2022visually,
  title={Visually-Aware Audio Captioning With Adaptive Audio-Visual Attention},
  author={Liu, Xubo and Huang, Qiushi and Mei, Xinhao and Liu, Haohe and Kong, Qiuqiang and Sun, Jianyuan and Li, Shengchen and Ko, Tom and Zhang, Yu and Tang, Lilian H and others},
  journal={arXiv preprint arXiv:2210.16428},
  year={2022}
}
```
