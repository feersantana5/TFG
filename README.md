# 3D Point Cloud Segmentation, Detection &amp; Classification by PointNet in a Low Cost System

![alt text](http://stanford.edu/~rqi/pointnet/images/teaser.jpg)

## System  

<img width="872" alt="image" src="https://user-images.githubusercontent.com/61494475/131353752-4b430586-bf7e-4a00-804b-b901f9a88ba0.png">

## Requirements ‼️
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

## Usage ⚙️
### Acquisition 📸

To capture 3D scenes using pmd Camboard pico flexx (MiData dataset):

    python 1.retrieveMiData.py
    
To capture 3D scenes using pmd Camboard pico flexx filtered by confidence value (MiDataCluster dataset):

    python 2.retrieveMiDataCluster.py

### Segmentation ✂️

To segment, filter and cluster the scene:

    python 3.miDataClusterSegmentation.py

### Train 🧠

To train and evaluate a PointNet model:

    python 4.pointNet.py

### Classification 🗂

To segment and classify objects in a scene:

    python 5.segmentationClassification.py

## Results 📊
### Acquisition 

Depth Image</br>
<img width="251" alt="image" src="https://user-images.githubusercontent.com/61494475/131347478-61369b18-599b-4588-90d0-3129da39d26b.png"> </br>

Point cloud </br>
<img width="399" alt="image" src="https://user-images.githubusercontent.com/61494475/131347664-5a0d3d70-1906-42f5-bfd2-94f42f966b7b.png"></br>

### Segmentation 
Different type of objects segmented</br>
<img width="259" alt="image" src="https://user-images.githubusercontent.com/61494475/131348744-7d3ebd08-d3b1-45be-80a7-78b5985f12c3.png">
<img width="183" alt="image" src="https://user-images.githubusercontent.com/61494475/131348784-c5862542-ad4c-4f86-8373-d8d21848321b.png">
<img width="144" alt="image" src="https://user-images.githubusercontent.com/61494475/131348995-d01fd409-3f59-4ece-bed0-dfe638058169.png">

### Train 
 
Confusion matrix </br>
<img width="384" alt="image" src="https://user-images.githubusercontent.com/61494475/131349213-9bc161f1-6424-47ee-a639-ec17894d4b44.png"></br>
Classification report</br>
<img width="253" alt="image" src="https://user-images.githubusercontent.com/61494475/131349255-c28ca94a-fe41-4423-b563-e37b61310aa7.png"></br>
Multiple Predictions </br>
<img width="402" alt="image" src="https://user-images.githubusercontent.com/61494475/131349323-066d1e28-c289-4f59-b245-7b3fdd8815c3.png"></br>
Single prediction </br>
<img width="308" alt="image" src="https://user-images.githubusercontent.com/61494475/131349347-8dc58226-b72b-4013-8cf4-7a81695ca010.png"></br>

### Classification 
Original point cloud</br>
<img width="244" alt="image" src="https://user-images.githubusercontent.com/61494475/131349926-a382255c-6a0d-439b-b644-b3468869efcf.png"></br>
Clusters in filtered original cloud</br>
<img width="290" alt="image" src="https://user-images.githubusercontent.com/61494475/131350025-634623ae-3551-442b-a89f-8389deac9cd5.png"></br>
Clusters in filtered point cloud</br>
<img width="268" alt="image" src="https://user-images.githubusercontent.com/61494475/131350011-4d98a82e-77fd-4080-8a98-eb006ea42555.png"></br>
Cluster obtained after segmentation</br>
<img width="111" alt="image" src="https://user-images.githubusercontent.com/61494475/131349983-2311e4ab-4fab-4c76-9033-094e7d3cddae.png"></br>
Sampled cluster</br>
<img width="160" alt="image" src="https://user-images.githubusercontent.com/61494475/131350046-60240952-4504-4570-8c37-f4704a646a08.png"></br>
Prediction</br>
<img width="192" alt="image" src="https://user-images.githubusercontent.com/61494475/131350080-111dd684-7d6b-4161-a6ec-0290ec50f2a6.png"></br>

## Bibliography 📖
Original PointNet implementation: https://github.com/charlesq34/pointnet </br>
Original Keras implementation: https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet.py
