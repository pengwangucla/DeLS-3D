
# DeLS-3D
The code for DeLS-3D of CVPR 2018 and the jounal paper we submitted to TPAMI, website: [Apolloscape Dataset](apolloscape.auto).

```
    @inproceedings{wang2018dels,
        title={DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map},
        author={Wang, Peng and Yang, Ruigang and Cao, Binbin and Xu, Wei and Lin, Yuanqing},
        booktitle={CVPR},
        pages={5860--5869},
        year={2018}
    }

    @article{huangwang2018_apolloscape,
        author    = {Xinyu Huang and
                   Peng Wang and
                   Xinjing Chen and
                   Qichuan Geng and
                   Dingfu Zhou and
                   Ruigang Yang},
        title     = {The ApolloScape Open Dataset for Autonomous Driving and its Application},
        journal   = {CoRR},
        volume    = {abs/1803.06184},
        year      = {2018}
    }

```

![](./fig/pipeline.png)


# Dependents

Our system depends on:

API that supporting rendering of 3D points to label maps and depth maps:
[apolloscape-api](https://github.com/ApolloScapeAuto/dataset-api).

Various API that supporting many 3D applications
[vis_utils](https://github.com/pengwangucla/vis_utils)

Tested with Ubuntu 14.04 for rendering

If you need to train:
We use the code from:
[imgaug]()


# Dataset
Each part of the dataset are including, `images`, `camera_pose`, `semantic_label`, `semantic_3D_points`, `split`. 

| Data | `images`, `camera_pose`, `split` | `semantic_label` | `semantic_3D_points` |
|:-:|:-:|:-:|:-:|
|`Zpark`| [Download]() | [Download]() | [Download]() |
|`Dlake`| [Download]() | [Download]() | [Download]() |

Notice Dlake data is a subset of official released data [road01_inst.tar.gz](http://apolloscape.auto/scene.html) where you may obtain corresponding instance segmentation, depth and poses also.
Here we release the set used in the paper. 

In addition, for semantic 3D points, we release a merged pcd file which is used for rendering label maps in the paper, 
we also have much denser point cloud that are separated stored for Dlake dataset. 

[comment]: # Download and unzip to folder `data`, perturbed poses are also provided for results reproducibility.
Download and unzip to folder `data`


# Testing
Notice the number of the pre-trained models could be slightly different than that from paper due to the randomness from perturbation of GPS/IMU, but it should be close.

`source.rc` Firstly, pull the dependents and source the corresponding root folders


## Download pre-trained models

| Data | `PoseCNN` | `PoseRNN` | `SegmentCNN` |
|:-:|:-:|:-:|:-:|
|Zpark| [Download]()| [Download]()| [Download]()|
|Dlake| [Download]()| [Download]()| [Download]()|

Download them and put under 'models/${Data}/' ${Data} is the corresponding dataset name.

run the following code for test and evaluation of pose_cnn, pose_rnn, and seg_cnn. 

```Notice 
1. the images has some area (face&plate) blurred due to security reasons, therefore the number output from our pretrained models are different from that in paper. You may need to retrain the model for better performance.

2. The test code does not have clipping pose inside road area as preprocessing indicated due to liscense issue
```


```python
python test_DeLS-3D.py --dataset zpark --pose_cnn pose_cnn-0000 --pose_rnn pose_rnn-0000 --seg_cnn seg_cnn-0000
```

Should get results close to the paper with a random pose simulation. Notice current pipeline is an offline version, since CNN and RNN are not connected, one may need to reimplement for online version.


# Training
To be updated.


# Note
I may not have enough time for solving all your issues, expect delay of reply. 
Please try your best to solve the problems.

Contact: wangpeng54@baidu.com

