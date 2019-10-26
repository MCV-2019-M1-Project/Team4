# Team 4

Members: Sergio Casas Pastor, Sanket Biswas, Josep Brugués i Pujolràs

### Week 3

The tasks for the second week include:

* Create 2D/3D blocks and multiresolution histograms image descriptors
* Test system using new histograms and query set from last week QS2-W1
* Detect and remove text from images
* Evaluate text detection using bounding boxes and IoU parameter
* Evaluate the retrieval system for QS1-W2 removing the text region
* Evaluate the retrieval system for QS2-W2 which contains more than one painting per image, removing background and text regions

Slides for Week 3 are available here: [Slides](https://drive.google.com/open?id=1HnHFoQNfw116Y6bx3lS0ndQQlz-D3-SC)

### How to run

```sh
main.py <querySetPath> <backgroundRemoval> <textRemoval> <textRemovalMethod> <k> 
```

The querySetPath parameter indicates the location of the query set images.

The backgroundRemoval parameter indicates if background has to be removed from the query set images. 
* True: background removal
* False: no background removal.

The textRemoval parameter indicates if the text has to be removed from the query set images. 
* True: text removal 
* False: no text removal.

The textRemovalMethod parameter indicates the method used for text detection:
* 1 : text detection based on color segmentation
* 2 : text detection based on morphology operations

The k parameter is used to indicate the number of top results that need to be saved.