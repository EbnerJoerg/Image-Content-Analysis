"""

Generate the image data set for the labeling seminar.
Feel free to adapt this script to generate special shapes, color ranges,... what

"""
#download opencv: pip install opencv-python #not python package names
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import cv2
from typing import Union, Tuple, List

RANDOM_SEED = 123


def reshape_and_pad(image: np.ndarray, target_size: Union[List[int], Tuple[int]]):
    """ 
    reshape an image and pad it while keeping the ratio.
    :param image: image that shall be reshaped
    :param target_size: size of the target image
    :return: reshaped and padded image
    """
    # input shapes
    input_shape = image.shape[:2]
    # calculate the width and height
    w_ratio = float(target_size[1]) / input_shape[1]
    h_ratio = float(target_size[0]) / input_shape[0]
    # take the smaller ratio to ensure the whole image fits in the new shape
    ratio = min(w_ratio, h_ratio)
    # calculate the new size
    new_size = tuple([int(x * ratio) for x in input_shape])

    # resize the image
    scaled_image = cv2.resize(image, (new_size[1], new_size[0]))

    # width and height differences
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]

    # image position within the new image
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # padding color
    padding_color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(scaled_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return new_image


def main(train):
    print("starting to create the data set for the instagram images")
    # shape of the output images you can scale this if you want
    target_shape = [256,256]
    # the columns that will be predicted
    #target_columns = ["median_Advertisement", "median_Body Visible", "median_Human Focus", "median_Nudity"]
    target_columns = ["image_Advertisement", "image_Body_Visible", "image_Human_Focus", "image_Nudity"]
    # the directory where the images are stored
    #input_data_directory = "../../../Data/images_train/"
    if train:
        input_data_directory = "data/images_train/"
    else:
        input_data_directory = "data/images_test/"
    # get the image files in the directory
    input_files = sorted([f for f in listdir(input_data_directory) if isfile(join(input_data_directory, f))])
    input_files = np.array(input_files)
    # potentially take only a subset during development phase
    # input_files = input_files[:30]

    # number of images
    n_images = len(input_files)
    print("found {} input files".format(n_images))

    # define the point to split for the test set
    test_split_point = int(n_images * 0.8)

    # permute the images in a reproducible way
    permutation = np.random.RandomState(seed=RANDOM_SEED).permutation(n_images)
    input_files = input_files[permutation]

    # shape of the image array
    array_shape = [n_images] + target_shape + [3]
    image_array = np.ndarray(shape=array_shape, dtype=np.uint8)

    # array for the targets
    #majority_df = pd.read_csv("../../../Data/summary_of_majorities.csv")
    if train:
        majority_df = pd.read_csv("data/stud_df_train.csv")
    else:
        majority_df = pd.read_csv("data/stud_df_test.csv")
    try:
        # If problems in CSV-File  
        subdata2 = majority_df[(majority_df['image_width'].isna() == True)] # 294 
        subdata2 = subdata2["image_path"].str.split(",", expand = True)
        subdata2[7] = subdata2[7] + subdata2[8]
        subdata2 = subdata2.drop(8, axis=1)
        subdata2.columns = majority_df.columns
        df = subdata2.append(majority_df, ignore_index=False)
        del subdata2
        df = df.loc[~df.index.duplicated(keep='first')]
        majority_df = df.sort_index()
        del df
    except:
        pass


    targets = np.zeros((n_images, 4))

    # loop over all images, load and reshape them
    for index, input_file in enumerate(input_files):
        if index % 100 == 0:
            print("processing image", index)
        # load the image
        image_path = os.path.join(input_data_directory, input_file)
        image = cv2.imread(image_path)
        # scale it and add it to the array
        scaled = reshape_and_pad(image, target_shape)
        image_array[index] = scaled

        # extract the target
        #part = majority_df[majority_df["image_name"] == "/" + input_file]
        part = majority_df[majority_df["image_path"] == input_file]
        target_row = part[target_columns].values[0]
        targets[index] = target_row
    if train:
        # save the created image array
        np.save("data/x_all256.npy", image_array)
        np.save("data/x_train256.npy", image_array[:test_split_point])
        np.save("data/x_valid256.npy", image_array[test_split_point:])
        np.save("data/names_for_x256.npy", input_files)
        np.save("data/y_all256.npy", targets)
        np.save("data/y_train256.npy", targets[:test_split_point])
        np.save("data/y_valid256.npy", targets[test_split_point:])
    else:
        np.save("data/x_test256.npy", image_array)
        np.save("data/y_test256.npy", targets)
    
    # read the files again to see if they were created correctly
    #output_files = ["x_all.npy", "x_train.npy", "x_valid.npy", "names_for_x.npy", "y_all.npy", "y_train.npy",
    #                "y_valid.npy"]
    #for f in output_files:
    #    a = np.load(f)
    #    print("reloaded {} shape {}".format(f, a.shape))


if __name__ == '__main__':
    main(True)
    main(False)
