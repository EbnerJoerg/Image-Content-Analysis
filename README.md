# Image-Content-Analysis

For this analysis 4070 images of the most followed instagram pages are taken and labeled regarding the advertisement, the body visibility, the human focus and the nudity on a scale from 1 to 5.
One stands for no advertisement and 5 for pure advertisement, etc.
<br> 90 percent of the images are for training, 10 percent for testing.

To download the images and the labeled data please click [here](https://drive.google.com/file/d/1jnLIgELTjBhMWkJlBAN13NXGeXQuTiXX/view?usp=sharing).

>**Requirements:** <br>
>pip install opencv-python <br>
>pip install tensorflow <br>
>pip install scikit.learn

Workflow: <br>
1. Run *1_preprocess_train_test_df.py* to save numpy arrays of the transformed images <br>
2. Run *2_modeling.py* for Data Augmentation, training and validation, and prediction of the model
