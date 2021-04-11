# Rice-Root-Segmentation-and-Keypoint-Detection
This repository provides implementation with training/testing codes of various Rice-Root-Segmentation-and-Keypoint-Detection in Keras. 

Authors : Liang Gong, Xiaofeng Du, Kai Zhu<sup>[1]</sup>.

[Shanghai Jiao Tong University](www.sjtu.edu.cn/)<sup>[1]</sup>.

## Some visualizations from pretrained models:

<table border="0" align="center" cellpadding="0" cellspacing="0">
  <tr>
    <td valign="top"><img height="600" width="150" src="https://github.com/KaiZhuhhhhhh/Rice-Root-Segmentation-and-Keypoint-Detection/blob/master/test/RootMask/_1.png">
    <img height="600" width="150" src="https://github.com/KaiZhuhhhhhh/Rice-Root-Segmentation-and-Keypoint-Detection/blob/master/test/RootMask/maskHeatmap_1.png">
    <img height="600" width="150" src="https://github.com/KaiZhuhhhhhh/Rice-Root-Segmentation-and-Keypoint-Detection/blob/master/test/RootMask/rootHeatmap_1.png">
</div></td>
  </tr>
</table>

## Usage
GetTrainSet.m: Get train set of patches(root patch and mask patch)
GetTrainSet_keyPoint.m: Get train set of patches(root patch, mask patch and heatmap patch, used for keypoint detection)
RootSeg_UnetResSe.ipynb/keyPointDetection.ipynb: Code for sematic segmentation/keypoint detection of rice
RootUnetResnet_version_1.model: Pretrained model of RootSeg_UnetResSe
getPrediction.m/getPrediction_KeyPoint.m: Merge the patches predicted from test patchs into mask/heatmap of root
Otsu.m: Use otsu method for segmentation
compare.py: Compare our method and Otsu
keypoint/generateHeatMap.py: Generate heatmap from keypoints
rice_root_key_point.json: Project of keypoint annotation(http://www.robots.ox.ac.uk/~vgg/software/via/via.html)

## Citation
If you found our work useful and used in your research, please cite the paper:
    
    @inproceedings{hu2018senet,
      title={Rice-Root-Segmentation-and-Keypoint-Detection},
      author={KaiZhu},
      journal={unspecified},
      year={2019}
    }
