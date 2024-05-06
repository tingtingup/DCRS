





## Deep coupled registration and segmentation of multimodal whole-brain images
We introduce a deep learning framework for joint registration and segmentation of multi-modal brain images. Under this framework, registration and segmentation tasks are deeply coupled and collaborated at two hierarchical layers. In the inner layer, we establish a strong feature-level coupling between the two tasks by learning a uniffed common latent feature representation. In the outer layer, we introduce a mutually supervised dual-branch network to decouple latent features and facilitate task-level collaboration between registration and segmentation.

https://github.com/tingtingup/DCRS/blob/main/framework.jpg

This is a Pytorch implementation
## Using the Code
### Requirements
This code has been developed under Python 3.7.12, PyTorch 1.10, and Ubuntu 20.04.
### Installation
In addition to the above libraries, the python environment can be set as follows:
```shell
conda create -n CMIR
conda activate CMIR
pip install SimpleITK 
pip install scipy scikit-image matplotlib pandans
pip install cv2
pip install 
```
### Pre-training and joint learning

The segmentation network(S), registration network(R) and feature extraction network(G) are first pretrained individually based on the introduced ESDR:
```shell
python pretrain_feature_extraction_network.py
python pretrain_registration_network.py
python pretrain_segmentation_network.py
```



### Joint training

In the fine-tuning phase, we concatenate the pre-trained networks G, S and R to form G-S and G-R branches, and jointly optimize two branches in an iterative manner 

```shell
python co_train.py
```
