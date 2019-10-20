# Team 4

Members: Sergio Casas Pastor, Sanket Biswas, Josep Brugués i Pujolràs

### Week 2

The tasks for the second week include:

* Create 2D/3D blocks and multiresolution histograms image descriptors
* Test system using new histograms and query set from last week QS2-W1
* Detect and remove text from images
* Evaluate text detection using bounding boxes and IoU parameter
* Evaluate the retrieval system for QS1-W2 removing the text region
* Evaluate the retrieval system for QS2-W2 which contains more than one painting per image, removing background and text regions

Slides for Week 2 are available here: [Slides](https://drive.google.com/open?id=1HnHFoQNfw116Y6bx3lS0ndQQlz-D3-SC)

### How to run

```sh
main.py <colorBase> <dimension> <querySetPath> <metric> <k> <backgroundRemoval> <textRemoval> <textRemovalMethod>
```

The available color bases for the histograms include:

* 1D
* LAB
* HSV
* BGR
* YCrCb

The dimension parameter indicates the dimension of histograms:
* 1D
* 2D
* 3D

The querySetPath parameter indicates the location of the query set images.
The metric parameter indicates the metric that is used for comparing the images. The following metrics are available:

* euclidean_distance
* l1_distance
* cosine_distance
* correlation
* chi_square
* intersection
* hellinger_distance
* bhattacharya_distance
* kl_divergence
 
The k parameter is used to indicate the number of top results that need to be saved.

The backgroundRemoval parameter indicates if background has to be removed from the query set images. 
* True: background removal
* False: no background removal.

The textRemoval parameter indicates if the text has to be removed from the query set images. 
* True: text removal 
* False: no text removal.

The textRemovalMethod parameter indicates the method used for text detection:
* 1 : text detection based on color segmentation
* 2 : text detection based on morphology operations
