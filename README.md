# Team 4

Members: Sergio Casas Pastor, Sanket Biswas, Josep Brugués i Pujolràs

### Week 1

The tasks for the first week include:

* Create histogram image descriptors for both the museum images and the query sets
* Implement similarity measures to compare the images
* Implement a retrieval system to get the K top results
* Evaluate the results using the MAP@k metric
* Background removal of images from QS2
* Evaluate the masks and the retrieval system for QS2

Slides for Week 1 are available here: [Slides](https://drive.google.com/file/d/12lkVgFkJs0ZWDThkyyTrXcP2W6iZ5FQz/view?usp=sharing)

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

The backgroundRemoval parameter indicates if background has to be removed from the query set images. True for background removal and False for no background removal.

The textRemoval parameter indicates if the text has to be removed from the query set images. True for text removal and False for no text removal.

The textRemovalMethod parameter indicates the method used for text detection:
* 1 : text detection based on color segmentation
* 2 : text detection based on morphology operations
