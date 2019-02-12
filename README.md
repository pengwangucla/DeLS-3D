
# DeLS-3D
The code for DeLS-3D of CVPR 2018, the [Apolloscape Dataset](apolloscape.auto) is extended after the paper.

```
 @inproceedings{wang2018dels,
   title={DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map},
   author={Wang, Peng and Yang, Ruigang and Cao, Binbin and Xu, Wei and Lin, Yuanqing},
   booktitle={CVPR},
   pages={5860--5869},
   year={2018}
 }
```


# Dataset
Each part of the dataset are including, `images`, `camera pose`, `semantic label`, `semantic 3D point cloud`, `split`
More data will be added.

`Zpark`: [Download Link]()
`Dlake`: [Download Link]()


# Code
`Data`: include the dataset configuration and camera paramters for each dataset. 

`Projection:` 
`PoseCNN`
`PoseRNN`
`SegmentCNN`


# Testing
Notice the number of the pre-trained models could be slightly different than that from paper due to the randomness from perturbation of GPS/IMU, but it should be close.
`test.py`

## Pre-trained models
`PoseCNN`: [Download Link]()
`PoseRNN`: [Download Link]()
`SegmentCNN`: [Download Link]()


# Training
To be updated.




