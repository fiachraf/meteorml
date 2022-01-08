import numpy as np
import math
import os

def add_zeros_row(image, top_or_bottom, num_rows_to_add):
    """ adds rows of zeros to either the top or bottom of the numpy array
        if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    #shape returns (num_rows, num_cols)
    # print(f"image_shape: {image_shape}")
    num_rows = image_shape[0]
    num_cols = image_shape[1]

    # print(f"num_rows: {num_rows}")
    # print(f"num_cols: {num_cols}")
    zero_rows = np.zeros((num_rows_to_add, num_cols))
    # print(f"np.shape(zero_rows): {np.shape(zero_rows)}")
    # print(f"zero_rows: {zero_rows}")

    if top_or_bottom == "top":
        new_image = np.vstack((zero_rows, image))
        return new_image
    elif top_or_bottom == "bottom":
        new_image = np.vstack((image, zero_rows))
        return new_image
    #return None which will cause an error if invalid inputs have been used
    return


def add_zeros_col(image, left_or_right, num__cols_to_add):
    """ adds columns of zeros to either the left or right of the numpy array
        if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    #shape returns (num_rows, num_cols)
    num_rows = image_shape[0]
    num_cols = image_shape[1]
    # print(f"num_rows: {num_rows}")
    # print(f"num_cols: {num_cols}")
    zero_cols = np.zeros((num_rows, num__cols_to_add))
    # print(f"np.shape(zero_cols): {np.shape(zero_cols)}")
    # print(f"zero_cols: {zero_cols}")


    if left_or_right == "left":
        new_image = np.hstack((zero_cols, image))
        return new_image
    elif left_or_right == "right":
        new_image = np.hstack((image, zero_cols))
        return new_image
    #return None which will cause an error if invalid inputs have been used
    return


def blackfill(image, leftover_top=0, leftover_bottom=0, leftover_left=0, leftover_right=0):
    """ will make the image square by adding rows or columns of black pixels to the image, as the image needs to be square to be fed into a Convolutional Neural Network(CNN)

    As I am giving the cropped images +20 pixels on all sides from the detection square edges, any meteors that occur on the edge of the frame in the fits file will not be centered so I am using the leftover terms, to add columns or rows for the edges that were cut off so that the meteor is centered in the image. This might badly affect the data and thus the CNN so it might need to be changed. However I think it might help as the added space around the meteor I hope will make sure that it includes the entire meteor trail which can hopefully help differentiate true detections from false detections

    When squaring the image it will also keep the meteor detection in roughly the center

    Resizing of the images to be all the same size will be done in another script using the keras.preprocessing module
    Could also do squaring of images using keras.preprocessing module I just thought doing it this might yield better results as it wont be stretched or distored in a potentially non-uniform way
    """

    if leftover_top > 0:
        image = add_zeros_row(image, "top", leftover_top)
    if leftover_bottom > 0:
        image = add_zeros_row(image, "bottom", leftover_bottom)
    if leftover_left > 0:
        image = add_zeros_col(image, "left", leftover_left)
    if leftover_right > 0:
        image = add_zeros_col(image, "right", leftover_right)

    if np.shape(image)[0] == np.shape(image)[1]:
        new_image = image[:,:]
        return new_image

    if np.shape(image)[0] < np.shape(image)[1]:
        rows_needed = np.shape(image)[1] - np.shape(image)[0]
        image = add_zeros_row(image, "top", math.floor(rows_needed/2))
        image = add_zeros_row(image, "bottom", math.ceil(rows_needed/2))
        new_image = image[:,:]
        return new_image

    if np.shape(image)[1] < np.shape(image)[0]:
        cols_needed = np.shape(image)[0] - np.shape(image)[1]
        image = add_zeros_col(image, "left", math.floor(cols_needed/2))
        image = add_zeros_col(image, "right", math.ceil(cols_needed/2))
        new_image = image[:,:]
        return new_image

    return


def search_dirs(search_term, chosen_dir=os.getcwd(), file_or_folder="", exact_match=False, search_subdirs=False): #extension=""):
    """
    searches directory for the search term and returns a list of objects containing the search term
    chosen_dir is the directory to search
    file_or_folder designates if it should search for files or folders, by default it will search for both
    exact_match designates if only exact matches should be found
    search_subdirs designates if it should recursively search subdirectories
    extension designates if it should search for only files with the matching extension

    """
    matching_entries = []

    if search_subdirs == True:
        for root, dirs, files in os.walk(chosen_dir):
            if file_or_folder == "" or file_or_folder == "folder":
                for dir in dirs:
                    if exact_match == True:
                        if dir == search_term:
                            matching_entries.append(os.path.join(root, dir))
                    elif exact_match == False:
                        if dir.find(search_term) != -1:
                            matching_entries.append(os.path.join(root, dir))
            if file_or_folder == "" or file_or_folder == "file":
                for file in files:
                    if exact_match == True:
                        if file == search_term:
                            matching_entries.append(os.path.join(root, file))
                    elif exact_match == False:
                        if file.find(search_term) != -1:
                            matching_entries.append(os.path.join(root, file))


    elif search_subdirs == False:
        contents_list = os.listdir(chosen_dir)
        for item in contents_list:
            if exact_match == True:
                if item == search_term:
                    if file_or_folder == "folder":
                        if os.path.isdir(os.path.join(chosen_dir, item)) == True:
                            matching_entries.append(item)
                    elif file_or_folder == "file":
                        if os.path.isfile(os.path.join(chosen_dir, item)) == True:
                            matching_entries.append(item)
                    elif file_or_folder == "":
                        matching_entries.append(item)

            elif exact_match == False:
                if item.find(search_term) != -1:
                    if file_or_folder == "folder":
                        if os.path.isdir(os.path.join(chosen_dir, item)) == True:
                            matching_entries.append(item)
                    elif file_or_folder == "file":
                        if os.path.isfile(os.path.join(chosen_dir, item)) == True:
                            matching_entries.append(item)
                    elif file_or_folder == "":
                        matching_entries.append(item)

    return matching_entries
