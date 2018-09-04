# Data_Augmentation
This is a data augmentation program for images

The reason and effectiveness of data agumentation can be seen at: "Perez L, Wang J. The effectiveness of data augmentation in image classification using deep learning[J]. arXiv preprint arXiv:1712.04621, 2017".

In the real world scenario, we may have a dataset of images taken in a limited set of conditions. But, our target application may exist in a variety of conditions, such as different orientation, location, scale, brightness etc. 
We account for these situations by training our neural network with additional synthetically modified data.

To run this program, simply set the folder for the original/label images in data_augment_from_mask.py. Then run it by: 
       python data_augment_from_mask.py
    

The augmentated images will be automatically generated.
