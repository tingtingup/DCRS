






## Using the Code
### Requirements
This code has been developed under Python 3.7.12, PyTorch 1.10, and Ubuntu 20.04.

In addition to the above libraries, the python environment can be set as follows:
```shell
conda create -n CMIR
conda activate CMIR
pip install SimpleITK 
pip install scipy scikit-image matplotlib pandans
pip install cv2
pip install 
```



### sdf-generator, the segmentation network and the alignment network are first pre-trained separately and then trained jointly

### ##pretrain sdf_generator
python pre_train_sdf_generator.py
### ##pretrain segmentation
python pretrain_segmentation.py
### ##pretrain registration
python pretrain_registration.py

### joint training
python co_train.py
