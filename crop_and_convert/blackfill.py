import numpy as np
import math
import os
import sys
import traceback
from pathlib import Path
<<<<<<< HEAD
#could be changed if I installed RMS properly but I don't want to install everythin for RMS on my personal laptop 
=======
#could be changed if I installed RMS properly but I don't want to install everythin for RMS on my personal laptop
>>>>>>> f2799bd495e7f547b9d62ab6a1862bd57d16526d
path_root = Path(__file__).parents[1]
#print(f"path_root: {path_root}/RMS")
sys.path.insert(1, str(path_root) + "/RMS")

from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo

def add_zeros_row(image, top_or_bottom, num_rows_to_add):
    """ adds rows of zeros to either the top or bottom of the numpy array
        if performance is important the same effect can be achieved using numpy slicing which is faster
    """
    image_shape = np.shape(image)
    #shape returns (num_rows, num_cols)
    num_rows = image_shape[0]
    num_cols = image_shape[1]

    zero_rows = np.zeros((num_rows_to_add, num_cols))

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
    zero_cols = np.zeros((num_rows, num__cols_to_add))


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


def search_dirs(search_term, chosen_dir, file_or_folder="", exact_match=False, search_subdirs=False): #extension=""):
    """
    searches directory for the search term and returns a list of objects containing the search term
    chosen_dir is the directory to search
    file_or_folder designates if it should search for files or folders, by default it will search for both
    exact_match designates if only exact matches should be found
    search_subdirs designates if it should recursively search subdirectories
    extension designates if it should search for only files with the matching extension

    Returns list of tuples, index 0 is the directory where the file is located and index 1 is the name of the file
    """
    matching_entries = []

    if search_subdirs == True:
        for root, dirs, files in os.walk(chosen_dir):
            if file_or_folder == "" or file_or_folder == "folder":
                for dir in dirs:
                    if exact_match == True:
                        if dir == search_term:
                            matching_entries.append((root, dir))
                    elif exact_match == False:
                        if dir.find(search_term) != -1:
                            matching_entries.append((root, dir))
            if file_or_folder == "" or file_or_folder == "file":
                for file in files:
                    if exact_match == True:
                        if file == search_term:
                            matching_entries.append((root, file))
                    elif exact_match == False:
                        if file.find(search_term) != -1:
                            matching_entries.append((root, file))


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
                            matching_entries.append((os.getcwd(), item))
                    elif file_or_folder == "file":
                        if os.path.isfile(os.path.join(chosen_dir, item)) == True:
                            matching_entries.append((os.getcwd(), item))
                    elif file_or_folder == "":
                        matching_entries.append((os.getcwd(), item))

    return matching_entries


def crop_detections(detection_info, fits_dir):
    """
    crops the detection from the fits file using the information provided from the FTPdetectinfo files
    detection_info is a single element of the list returned by the RMS.RMS.Formats.FTPdetectinfo.readFTPdetectinfo() function. This list contains only information on a single detection
    fits_dir is the the directory where the fits file is located

    returns the cropped image as a Numpy array
    """

    fits_file_name = detection_info[0]
    meteor_num = detection_info[2]
    num_segments = detection_info[3]
    first_frame_info = detection_info[11][0]
    first_frame_no = first_frame_info[1]
    last_frame_info = detection_info[11][-1]
    last_frame_no = last_frame_info[1]

    try:
        #read the fits_file
        # print(f"fits_dir: {fits_dir}\nfits_file_name: {fits_file_name}")
        fits_file = FFfile.read(fits_dir, fits_file_name, fmt="fits")
        #image array with background set to 0 so detections stand out more
        #TODO inlcude code to use mask for the camera, currently masks not available on the data given to me, Fiachra Feehilly (2021)
        detect_only = fits_file.maxpixel - fits_file.avepixel
        #set image to only include frames where detection occurs, reduces likelihood that there will then be multiple detections in the same cropped image
        detect_only_frames = FFfile.selectFFFrames(detect_only, fits_file, first_frame_no, last_frame_no)

        #get size of the image
        row_size = detect_only_frames.shape[0]
        col_size = detect_only_frames.shape[1]

        #side 1, 2 are the left and right sides but still need to determine which is which
        # left side will be the lesser value as the value represents column number
        side_1 = first_frame_info[2]
        side_2 = last_frame_info[2]
        if side_1 > side_2:
            right_side = math.ceil(side_1) + 1 #rounds up and adds 1 to deal with Python slicing so that it includes everything rather than cutting off the last column
            left_side = math.floor(side_2)
        else:
            left_side = math.floor(side_1)
            right_side = math.ceil(side_2) + 1
        #side 3 and 4 are the top and bottom sides but still need to determine which is which
        # bottom side will be the higher value as the value represents the row number
        side_3 = first_frame_info[3]
        side_4 = last_frame_info[3]
        if side_3 > side_4:
            bottom_side = math.ceil(side_3) + 1
            top_side = math.floor(side_4)
        else:
            top_side = math.floor(side_3)
            bottom_side = math.ceil(side_4) + 1

        #add some space around the meteor detection so that its not touching the edges
        #leftover terms need to be set to 0 outside if statements otherwise they wont be set if there's nothing left over which will cause an error with the blackfill.blackfill() line
        left_side = left_side - 20
        leftover_left = 0
        if left_side < 0:
            #this will be used later to determine how to fill in the rest of the image to make it square but also have the meteor centered in the image
            leftover_left = 0 - left_side
            left_side = 0

        right_side = right_side + 20
        leftover_right = 0
        if right_side > col_size:
            leftover_right = right_side - col_size
            right_side = col_size

        top_side = top_side - 20
        leftover_top = 0
        if top_side < 0:
            leftover_top = 0 - top_side
            top_side = 0

        bottom_side = bottom_side + 20
        leftover_bottom = 0
        if bottom_side > row_size:
            leftover_bottom = bottom_side - row_size
            bottom_side = row_size


        #get cropped image of the meteor detection
        #first index set is for row selection, second index set is for column selection
        crop_image = detect_only_frames[top_side:bottom_side, left_side:right_side]
        square_crop_image = blackfill(crop_image, leftover_top, leftover_bottom, leftover_left, leftover_right)

        # #this bit is only needed to plot the image for visual demonstrations
        # #-------------------------------------------------------------------
        # #create plot of meteor_image and crop_image
        # # has full image displayed on left side and then the can do two cropped images displayed on the right side
        # fig, axd = plt.subplot_mosaic([['big_image', 'big_image', 'crop_image'],
        #                                 ['big_image', 'big_image', 'small_image_2']],
        #                                 )
        #
        # axd['big_image'].imshow(detect_only_frames)
        # axd['crop_image'].imshow(crop_image)
        # axd['small_image_2'].imshow(square_crop_image)
        #
        # # Create a Rectangle patch Rectangle((left column, top row), column width, row height, linewidth, edgecolor, facecolor)
        # rect = patches.Rectangle((left_side, top_side), (right_side - left_side), (bottom_side - top_side), linewidth=1, edgecolor='r', facecolor='none')
        #
        # # Add the patch to the big image
        # axd['big_image'].add_patch(rect)
        #
        # # change the size of the figure
        # fig.set_size_inches(18.5, 10.5)
        #
        # # display the plot
        # plt.show()
        #
        # #-------------------------------------------------------------------

        return square_crop_image

    except Exception as error:
        print(f"error: {traceback.format_exc()}")
        return None
