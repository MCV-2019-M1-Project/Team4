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
main.py <colorBase> <querySetPath> <metric> <k> <backgroundRemoval>
```

The available color bases for the histograms include:

* 1D
* LAB
* HSV
* BGR
* YCrCb

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
Finally, the backgroundRemoval parameter indicates if background has to be removed from the query set images. 0 for no backgroud removal, 1 for the first method and 2 for the second method.
