# Kaggle Competition
# UW-Madison GI Tract Image Segmentation


## Competition Information

PRIMARY TASK DESCRIPTION

In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.


BASIC BACKGROUND INFORMATION

In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day.

In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.

## Notebooks overview

In this competition we are segmenting organs cells in images. The medical images consist in MRI scans, where **Stomach**, **Large Bowel** & **Small Bowel** might be present. It is a **MultiLabel Segmentation** task as all classes might be present in one image.
There are 2 notebooks
* uwm-gi-tract-segmentation-eda-only.ipynb: An EDA notebook where the dataset structure and content is explored
* uwm-transunet-2-5d-training-tf.ipynb: A Training notebook where a TransUNet model is trained on the dataset

## Segmentation Methodology and Model
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
