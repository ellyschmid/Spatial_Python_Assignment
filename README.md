# Spatial_Python_Assignment
In this project, an RCNN-Model (regional-convolutional-neural-network) is used for ship detection. A Sentinel-1 Scene from the Suez Canal was used as the training image, using the 95th percentile of the sentinel backscatter image in VH polarization, which improves the visibility of ships.
![grafik](https://github.com/ellyschmid/Spatial_Python_Assignment/assets/116875590/2ee65455-b705-4a1b-9479-b862b410b4f6)

## The Project consits of 4 Scripts 
### 1. clip_images.py 
This code takes the Sentinel Image, that was previously converted from .tif to .png and clips it using the ship polygons of a shapefile. These clipped images are then saved as RGB converted versions in PNG files. In summary, this code snippet utilizes geospatial data processing to extract smaller image segments focused on areas containing ships. These images are subsequently employed in training the model.

### create_annotation_files.py
This script serves as an interactive annotation tool designed specifically for identifying and annotating regions of interest containing objects within images. It employs the matplotlib library to create an interactive environment that enables users to draw bounding boxes around the areas of images where the deired objects are present. These bounding boxes serve as annotations for machine learning tasks such as object detection. The script generates XML annotations for each bounding box, encapsulating essential information about the objects location and image attributes. 

### convert_xml_to_csv,py
Converts XML annotations to CSV format for improved data utilization

### rcnn_ship_detection.py 
This code demonstrates the process of ship detection using RCNN. It involves preprocessing images and annotations, creating and training a RCNN model (based on VGG16), performing data augmentation, and utilizing selective search to identify and visualize regions of interest (potential ships) in the test image. The code showcases model training, evaluation, and testing for identifying whether an image contains ships or not.

### Provided Data
1. already clipped Images containing single ships for training (ships_images.zip folder)
2. and the corresponding annotation files in csv format (annotations_csv.zip folder)
3. the shapefile containing the ship polygons (ships.zip folder)

## Result 
![grafik](https://github.com/ellyschmid/Spatial_Python_Assignment/assets/116875590/2a976017-d408-43a5-9064-834c01c16707)

The result shows that the model successfully identifies ships using RCNN object detection on Sentinel-1 backscatter images.

### References 
- code template for RCNN-Model based on https://github.com/Hulkido/RCNN
