# Kaggle Competition
# UW-Madison GI Tract Image Segmentation


## Description

Track healthy organs in medical scans to improve cancer treatment. 

In this competition we are segmenting organs cells in images. The medical images consist in MRI scans, where **Stomach**, **Large Bowel** & **Small Bowel** might be present. It is a **MultiLabel Segmentation** task as all classes might be present in one image.
There are 2 notebooks
* An EDA notebook where the dataset structure and content is explored
* A Training notebook where a TransUNet model is trained on the dataset

## Methodology 
* In this notebook **2.5D** images are used for Training for **Segmentation** with `tf.data`, `tfrecord` using `Tensorflow`.  
* **2.5D Image Training** is training of **3D** image like **2D** Image. 2.5D images can take leverage of the extra depth information like our typical RGB image. 2.5D Images are built from 3 channels with 2 strides 
* The TransUNet model from **[TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)** is used here (from the transunet library).
* The model has 100M parameters that we need to train. To use TPU capabilities, the dataset has to be transformed into a TFRecord. I used the 2.5D image dataset created in this notebook by awsaf49: [UWMGI: 2.5D TFRecord Data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-tfrecord-data).
* "TFRecord files are created using **StratifiedGroupFold** to avoid data leakage due to `case` and to stratify `empty` and `non-empty` mask cases".
* This notebook is compatible for both **GPU** and **TPU**. Device is automatically selected so you won't have to do anything to allocate device.
* As there are overlaps between **Stomach**, **Large Bowel** & **Small Bowel** classes, this is a **MultiLabel Segmentation** task, so final activaion should be `sigmoid` instead of `softmax`.


### Dependencies

* Libraries Transunet and Segmentation-models are used to build the Unet


## Authors

Maud Comboul
https://github.com/maud-em/
