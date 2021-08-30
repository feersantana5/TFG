# 3D Point Cloud Segmentation, Detection &amp; Classification by PointNet

![alt text](http://stanford.edu/~rqi/pointnet/images/teaser.jpg)

## Requirements
* Python 3.6
* Open3D
* libroyale
* Tensorflow
* Keras
* Trimesh
* Scikit-learn
* Seaborn
* NumPy
* Matplotlib

## Usage
### Acquisition

To capture 3D scenes using pmd Camboard pico flexx:

    python 1.retrieveMiData.py
    
To capture 3D scenes using pmd Camboard pico flexx filtered by confidence value:

    python 2.retrieveMiDataCluster.py

### Segmentation

To segment, filter and cluster the scene:

    python 3.miDataClusterSegmentation.py

### Train

To train and evaluate a PointNet model:

    python 4.pointNet.py

### Classification

To segment and classify objects in a scene:

    python 5.segmentationClassification.py

## Bibliography 
Original PointNet implementation: https://github.com/charlesq34/pointnet </br>
Original Keras implementation: https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet.py
