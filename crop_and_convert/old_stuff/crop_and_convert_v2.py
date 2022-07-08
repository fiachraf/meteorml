from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os
import sys
import fiachra_python_logger as logger
import datetime
from PIL import Image
import traceback
import blackfill

#sys path needs to be changed depending on machine or just have RMS added properly as a package
sys.path.insert(1, "/home/fiachra/atom_projects/meteorml/RMS")

from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo



#file paths handled by input from user

#to get current working directory
#home_dir = os.getcwd()

#log command: (custom logger, just wanted something simple that I could look back on afterwards, could easily be changed out or removed)
#log(log_file_name="log.csv", log_priority, log_type, logger_call_no, details, log_time=datetime.datetime.now()):

cwd = input("directory path which has ConfirmedFiles and RejectedFiles folders: ")  #Meteor_Files directory path e.g. ~/Downloads/Meteor_Files
os.chdir(cwd)

home_dir = os.getcwd()  #Meteor_Files directory
print(home_dir)

top_directories = os.listdir()  #should be ConfirmedFiles and RejectedFiles folders amd Empty directories too
print(f"top_directories: {top_directories}")

#maybe add line for input to confirm that user has inputted the correct directory, otherwise get them to cancel the script

#creating new directories for the png versions of ConfirmedFiles and RejectedFiles
if "ConfirmedFiles_png" not in top_directories or "RejectedFiles_png" not in top_directories:
    try:
        os.mkdir("ConfirmedFiles_png")
    except:
        pass
    try:
        os.mkdir("RejectedFiles_png")
    except:
        pass

#go through the folders in the top_directories and then only go through the ConfirmedFiles and RejectedFiles folders
for dir_name in top_directories:
    os.chdir(home_dir)
    cwd = os.getcwd()

    try:
        if dir_name == "ConfirmedFiles" or dir_name == "RejectedFiles" and os.path.isdir(dir_name) == True:    #if not directory containing empty directories

            os.chdir(home_dir +"/" + dir_name)    #now in either ConfirmedFiles or RejectedFiles
            cwd = os.getcwd()
            print("cwd1=", cwd)
            detection_folder_list = os.listdir() #should be BE0001....5, BE0001....6, etc.

            try:
                for folders in detection_folder_list:   #folders is a folder which is a single night of detections from an operating station

                    try:
                        print("\n")
                        # print("cwd2", cwd)
                        os.chdir(home_dir + "/" + dir_name + "/" + folders)
                        cwd = os.getcwd()
                        print("cwd3", cwd)

                        files_list = os.listdir()   #individual files in the detection folder, e.g. BE0001....fits, FTPdetectinfo....txt etc.

                        #list of fits files that are not anlaysed as they dont appear in the FTPdetectinfo file
                        # files_not_analysed = files_list   #this operation simply gives the same list two different names so changes to one will change the other FML
                        files_not_analysed = files_list[:] #this is the fastest method to make an actual copy, other methods exist for different scenarios when the list objects aren't strings
                        #elements from list are removed later

                        #list of fits files that dont exist in the folder but are listed in the FTPdetectinfo file
                        fits_dont_exist = []
                        #both lists will be used to log files which have been missed or just don't exist
                        #these files should be manually checked just in case there are naming discrepancies or problems with my code


                        #as some folders have multiple FTPdetectinfo files, need to sort through them and pick correct one
                        FTPdetect_list = []
                        for item in files_list:
                            #search through files for FTPdetectinfo file, might be multiple files or file named differently
                            if item.find("FTPdetectinfo") != -1:
                                FTPdetect_list.append(item)

                        if len(FTPdetect_list) == 0:
                            print("no FTPdetectinfo file found")
                            sleep(2)
                            logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "FTPdetectinfo File not found", logger_call_no = 1, details = f"failed to find FTPdetectinfo file in dir: {cwd}")

                        else:

                            chosen_FTP_file = ""

                            print("\n")
                            for index, file in enumerate(FTPdetect_list):
                                # print(f"file[:-4]: {file[:-4]}")
                                # if an FTPdetectinfo file name in the folder matches the naming scheme (2 letters at the start, nothing like pre-confiramtion appended to the end) then it will automatically choose that FTPdetectinfo_ file
                                if file == "FTPdetectinfo_" + folders + ".txt":
                                    chosen_FTP_file = file
                                    print(f"13, FTPdetect_list: {FTPdetect_list}")
                                    print(f"27, chosen_FTP_file: {chosen_FTP_file}")

                            #if no FTPdetectinfo auto chosen then, print options available and have the user choose the correct one
                            if chosen_FTP_file == "":
                                print("\n choose best FTPdetectinfo file")
                                for possible_file_index, possible_file in enumerate(FTPdetect_list):
                                    print(f"file_name: {possible_file}, index: {possible_file_index}")

                                #take first input and if bad input (empty, not a number, out of range) is inputted then prompt user again
                                while True:
                                    try:
                                        chosen_FTP_file_index = int(input("please enter index of FTPdetectinfo file to be used to pick out meteors from the image: "))
                                        if chosen_FTP_file_index >= 0 and chosen_FTP_file_index < len(FTPdetect_list):
                                            chosen_FTP_file = FTPdetect_list[chosen_FTP_file_index]
                                            break

                                    except Exception as error_4:
                                        print("non number value entered, try again")


                            #read data from FTPdetectinfo file
                            FTP_file = FTPdetectinfo.readFTPdetectinfo(cwd, chosen_FTP_file)

                            #loop through each image entry in the FTPdetectinfo file and analyse each image
                            for detection_entry in FTP_file:
                                fits_file_name = detection_entry[0]

                                try:
                                    #if the fits file mentioned in the FTPdetectinfo file doesn't ecist then its name is added to the fits_dont_exist list which is then logged
                                    if fits_file_name not in files_list:
                                        fits_dont_exist.append(fits_file_name)

                                    #otherwise if the fits file does exist
                                    else:

                                        #read the fits_file
                                        fits_file = FFfile.read(cwd, fits_file_name, fmt="fits")

                                        #image array with background set to 0 so detections stand out more
                                        #TODO inlcude code to use mask for the camera, currently masks not available on the data given to me, Fiachra Feehilly (2021)
                                        detect_only = fits_file.maxpixel - fits_file.avepixel

                                        num_segments = detection_entry[3]
                                        first_frame_info = detection_entry[11][0]
                                        last_frame_info = detection_entry[11][-1]
                                        meteor_num = detection_entry[2]
                                        first_frame_no = first_frame_info[1]
                                        last_frame_no = last_frame_info[1]

                                        #accomodates for multiple meteor detections in one image otherwise files_not_analysed.remove(fits_file_name) would cause an error with multiple error detections
                                        if meteor_num > 1:
                                            files_not_analysed.append(fits_file_name)
                                        files_not_analysed.remove(fits_file_name)

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
                                        square_crop_image = blackfill.blackfill(crop_image, leftover_top, leftover_bottom, leftover_left, leftover_right)

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

                                        #save the Numpy array as a png using PIL
                                        im = Image.fromarray(square_crop_image)
                                        im = im.convert("L")    #converts to grescale
                                        im.save(home_dir + "/" + dir_name + "_png" + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")

                                        #don't think is necessary and will only fill up the log file, would be necessary if script doesn't finish and needs to be run again but would need to add additional code to have the script pick up where it had left off
                                        #logging files that have been analysed
                                        # logger.log(home_dir + "/log.csv", log_priority = "Low", log_type = "File processed", logger_call_no = 3, details = f"processed file:{fits_file_name}  in dir :{cwd}")

                                #logging any unanticipated errors that may occur
                                except Exception as error_5:
                                    logger.log(home_dir + "/log.csv", log_priority="High", log_type="file analysis failure", logger_call_no=10, details= traceback.format_exc())
                                    print(traceback.format_exc())

                            for item in files_not_analysed:
                                if item[-4:] != "fits":
                                    continue
                                logger.log(home_dir + "/log.csv", log_priority = "Medium", log_type = "File not analysed", logger_call_no = 4, details = f"this file: {item} in dir: {cwd} was not analysed")
                            for item in fits_dont_exist:
                                logger.log(home_dir + "/log.csv", log_priority = "Medium", log_type = "FITS File not found", logger_call_no = 5, details = f"failed to find FITS file: {item} in dir: {cwd}")

                    except Exception as error_3:
                        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 9, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                        print(f"error_3: {traceback.format_exc()}")

            except Exception as error_2:
                logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 8, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                print(f"error_2: {traceback.format_exc()}")

    except Exception as error_1:
        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 7, details = f"Error: {error_1} - occured in dir: {cwd}")
        print(f"error_1: {traceback.format_exc()}")

print("Cropping Script Finished")
logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Script finished", logger_call_no = 6, details = f"Script has finished")
