# Team 4

Members: Sergio Casas Pastor, Sanket Biswas, Josep Brugués i Pujolràs

### Week 3

The tasks for the second week include:

* Filter noise with linear or non-linear filters.
* On denoised images, detect box with overlapping text, and apply OCR to get the text. Test query system using QSD1-W2 using only text.
* Implement texture descriptors, and test them using QSD1-W2 using only texture descriptors.
* Combine descriptors (text + color, text + texture, texture + color, text + color + texture) and test the retrieval on QSD1-W3.

* Repeat the previous analysis with QSD2-W3 (remove noise, remove background, find 1 or 2 paintings and return the correspondences.


Slides for Week 3 are available here: [Slides](https://drive.google.com/open?id=1oaOwgYm6ZvufTF3qXziYM_mJLN3E2yK4jtHCWR7Y84Q)

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
