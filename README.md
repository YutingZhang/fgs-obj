Introduction
============
This software is the demo code for the following paper:

- Yuting Zhang, Kihyuk Sohn, Ruben Villegas, Gang Pan, Honglak Lee, “**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**”, *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015.

Please cite the above paper, if you use this software for your publications. 


Setup
=====

Installation
------------

The code can be obtained at

-   <https://github.com/YutingZhang/fgs-obj>

Run the following bash script in the root folder to download necessary dependency, including the Selective Search [1], GPML [2], minFunc [3] toolboxes:

        $ ./get_dependency.sh

Run the following bash script in the root folder of the code to get the trained models on PASCAL VOC2007 dataset:

        $ ./get_data.sh

After that, run the following bash script to set up the trained models for the test demo by

        $ ./setup_voc2007_models.sh

Compilation
-----------

The code is mainly in `MATLAB`, and depends on several external toolboxes. Most non-`MATLAB` code can be compiled automatically when you run the demo code for the first time except for the Caffe toolbox [1]. A customized version of Caffe toolbox is included in “`./caffe`”. This version is based on an official version release in September, 2014. The main modifications we made in addition to the original Caffe toolbox are as follows:

-   An enhanced `MATLAB` wrapper: more functions to interact with the Caffe backend.

-   A modified window \_data\_layer, termed fast\_window\_data\_layer, to fit our data format and to give better data loading efficiency.

-   Command line interfaces with slightly different argument lists.

Please refer to the following instruction to compile the Caffe toolbox
and its `MATLAB` wrapper:

-   <http://caffe.berkeleyvision.org/installation.html>

Basically, you need to success in running

        $ make
        $ make matcaffe

The other external dependencies should be compiled automatically when you first run the demo. If not, please do it by yourself. You can find these toolboxes inside the subfolder “`./dependency`”.

The code is tested on `MATLAB R2014a`. It should also work with older but recent versions of `MATLAB` (e.g., `R2013b`).

Testing with the trained models
===============================

The code contains the trained detection models. In this section, we addresses how to run the testing demo with the existing models.


Visualization demo on a single image
------------------------------------

### Run quick demo with one-line code

We can run detection on a new image as follows:

        >> simple_demo4(image_matrix/image_file_name, [model_type]);

where `model_type='struct'/'linear'` specifies which classifier to use.
If `model_type` is omitted:

-   if `simple_demo4` is called for the first time, `model_type='struct'` by default

-   otherwise, `model_type` is kept the same as the previous call.

This is one example:

        >> simple_demo4('000220.jpg');

You will see two figures on the screen: the first one is for the detection results without FGS, and the second one is for the whole pipeline (i.e., with FGS).

### Get the boxes and scores

You can obtain the bounding box coordinates and detection scores. First,
load the model by

        >> det_model = detInit;

You don’t need to run the command every time as long as it is in the
workspace. Then, use the following command do detection:

        >> [boxes,scores] = detSingle(I, det_model, use_gp, thresh);

where

-   `I` is an image matrix loaded by `imread` function (e.g.,
    `>> I = imread('000220.jpg');`).

-   `use_gp = 0/1` indicates whether to run the GP-based FGS.

-   `thresh`: you can set high threshold value to get less number of
    bounding boxes, and vice versa. By default, it is set 0.

-   `boxes` and `scores` are 20 dim cell array. Each cell corresponds to
    one object category defined in PASCAL VOC2007.

After this procedure, you can show the boxes by:

        >> detSingle(I, boxes, scores, det_model, thresh);

Here, you have an option to use higher threshold to rule out false
positives.

Benchmark on PASCAL VOC 2007
----------------------------

Run the follow command to generate the benchmark results for all the 20 categories of objects on PASCAL VOC2007:

        >> benchmark_voc2007(model_type, use_gp)

where `model_type='struct'/'linear'` indicates whether to use the linear SVM or the structured SVM classifier, and `use_gp=0/1` indicates whether to use the Gaussian process (GP) based fine-grained search (FGS).
There are four possible combinations of methods as in the following table.

| Methods          | BBoxReg | `model_type` | `use_gp` |
| ---------------- |---------|--------------|----------|
| R-CNN (VGGNet)   |   Yes   |  `'linear'`  |   `0`    | 
| + StructObj      |   Yes   |  `'struct'`  |   `0`    |
| + FGS            |   Yes   |  `'linear'`  |   `1`    |
| +StructObj + FGS |   Yes   |  `'struct'`  |   `1`    |

It will be very time-consuming to generate the results from scratch, if
you do it with only one on one GPU. Please distribute the jobs to multiple GPUs by yourself. 

When “`benchmark_voc2007`” is done, the precision, recall and the
average precision (AP) for each object category will be displayed on the
screen and saved into a mat file.

Model Training 
===============

The training procedure consists of multiple stages. The intermediate
outputs of all the stages will be cached. Please remove the cached
files in the folder “`./voc2007_train_cache`” to enable retraining from
the very scratch. If you do not want to retrain the CNN model, please leave the “`./voc2007_train_cache/CaffeModel`” untouched, and remove only other subfolders.

CNN fine-tuning
---------------

Suppose the the folder “`./voc2007_train_cache`” is empty. To finetune
the pretrained CNN model on the PASCAL VOC2007 database, you run the
following commands in `MATLAB`:

        >> trainInit_svmLinear
        >> trainCallStage('BoxList4Finetune');

You will get the list of bounding boxes that is required to finetune the
CNN under the following folder:

        ./voc2007_train_cache/BoxList4Finetune

Then, run the following script in **bash** to finetune the pretrained
CNN model:

        $ ./finetune_vgg16_on_voc2007.sh [gpu_id(0 as default)]

Your GPUs should have at least 12GB memory (e.g., K40c) to do the
finetuning. There are several CNN models pretrained on ImageNet that are
publicly available, and the 16-layer CNN model that we used in the experiment can be downloaded from the following:
<https://gist.github.com/ksimonyan/211839e770f7b538e2d8>

Learning detection models
-------------------------

With the CNN model finetuned on PASCAL VOC2007, you can train the
detection models by running the following command in `MATLAB`:

        >> train_demo(model_type);

where `model_type = 'struct'/'linear'`. This indicates whether to use the linear SVM or the structured SVM for the classifier. The stages shared between the two cases will run only once if you want to train both the two models. We include a stage-by-stage description of the training procedure in “`train_demo.m`”.

References
==========
[1] Caffe toolbox: <http://caffe.berkeleyvision.org/>

[2] Selective search toolbox:
<http://homepages.inf.ed.ac.uk/juijling/#page=projects1>

[3] GPML toolbox: <http://www.gaussianprocess.org/gpml/code/matlab/doc/>

[4] minFunc toolbox: <http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>

