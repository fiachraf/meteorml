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

#file paths need to be changed depending on machine
# ConfirmedFiles = "/home/fiachra/Downloads/Meteor_Files/ConfirmedFiles"
# RejectedFiles = "/home/fiachra/Downloads/Meteor_Files/RejectedFiles"

#to get current working directory
#home_dir = os.getcwd()

#log command
#log(log_file_name="log.csv", **log_priority, **log_type, **code_line, **details, log_time=datetime.datetime.now()):

cwd = input("directory path which has ConfirmedFiles and RejectedFiles folders: ")  #Meteor_Files directory path e.g. ~/Downloads/Meteor_Files
os.chdir(cwd)

home_dir = os.getcwd()  #Meteor_Files directory
print(home_dir)

top_directories = os.listdir()  #should be ConfirmedFiles and RejectedFiles folders amd Empty directories too
print(f"top_directories: {top_directories}")

if "ConfirmedFiles_png" not in top_directories or "RejectedFiles_png" not in top_directories:
    try:
        os.mkdir("ConfirmedFiles_png")
    except:
        pass
    try:
        os.mkdir("RejectedFiles_png")
    except:
        pass

for dir_name in top_directories:
    os.chdir(home_dir)
    cwd = os.getcwd()

    try:
        if dir_name.find("Empty") == -1 and dir_name.find("_png") == -1 and os.path.isdir(dir_name) == True:    #if not directory containing empty directories

            os.chdir(home_dir +"/" + dir_name)    #now in either ConfirmedFiles or RejectedFiles
            cwd = os.getcwd()
            print("cwd1=", cwd)


            detection_folder_list = os.listdir() #should be BE0001....5, BE0001....6, etc.

            try:
                for folders in detection_folder_list:

                    try:
                        print("\n")
                        #os.chdir(dir_name) #maybe need this one
                        print("cwd2", cwd)
                        os.chdir(home_dir + "/" + dir_name + "/" + folders)
                        cwd = os.getcwd()
                        print("cwd3", cwd)

                        files_list = os.listdir()   #individual files in the detection folder, e.g. BE0001....fits, FTPdetectinfo....txt etc.

                        #list of fits files that are not anlaysed as they dont appear in the FTPdetectinfo file
                        # files_not_analysed = files_list   #this operation simply gives the same list two different names so changes to one will change the other FML
                        files_not_analysed = files_list[:] #this is the fastest method to make an actual copy, other methods exist for different scenarios when the list objects aren't strings
                        # print(f"files_not_analysed: {files_not_analysed}")
                        #elements from list are removed later
                        #list of fits files that dont exist in the folder but are listed in the FTPdetectinfo file
                        fits_dont_exist = []
                        #both lists will be used to log files which have been missed
                        #these files need to be manually checked just in case there are naming discrepancies or problems with my code



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
                                # print(f"11, FTPdetect_list: {FTPdetect_list}")
                                # print(f"111, file[:-4]: {file[:-4]}")
                                # if file[-4:] != ".txt":
                                #     FTPdetect_list.remove(file)
                                #     print(f"12, FTPdetect_list: {FTPdetect_list}")

                                # print("test1, FTPdetectinfo_ + folders:", "FTPdetectinfo_" + folders)
                                # print("test2, FTPdetectinfo_ + folders[:-9]:", "FTPdetectinfo_" + folders[:-9])
                                print(f"file[:-4]: {file[:-4]}")
                                if file[:-4] == "FTPdetectinfo_" + folders or file[:-4] == "FTPdetectinfo_" + folders[:-9]:
                                    chosen_FTP_file = file
                                    print(f"13, FTPdetect_list: {FTPdetect_list}")
                                    print(f"27, chosen_FTP_file: {chosen_FTP_file}")
                            # print(f"21, FTPdetect_list: {FTPdetect_list}")
                            if chosen_FTP_file == "":
                                print("\n choose best FTPdetectinfo file")
                                for possible_file, possible_file_index in enumerate(FTPdetect_list):
                                    # print(f"14, FTPdetect_list: {FTPdetect_list}")

                                    print(f"file_name: {possible_file}, index: {possible_file_index}")

                                while True:
                                    try:
                                        chosen_FTP_file_index = int(input("please enter index of FTPdetectinfo file to be used to pick out meteors from the image: "))
                                        if chosen_FTP_file_index >= 0 and chosen_FTP_file_index < len(FTPdetect_list):
                                            chosen_FTP_file = FTPdetect_list[chosen_FTP_file_index]
                                            break

                                    except Exception as error_4:
                                        print("non number value entered, try again")


                            #opens detection file which lists all the detections in the images in its folder
                            with open(chosen_FTP_file, "r") as detect_file:

                                #reads the entire file into a list of strings
                                detect_file_lines_list = detect_file.readlines()

                                #searches each line for the string ".fits", if it finds it then it goes to the line which has the number of frames it was detected on and gets the the number of frames it was detected on
                                for line_number, line in enumerate(detect_file_lines_list):
                                    if line.find(".fits") != -1:
                                        # print(f"line: {line}")
                                        # print(f"type(line): {type(line)}")
                                        if line[:-1] not in files_list:
                                            fits_dont_exist.append(line[:-1])
                                        else:

                                            info_line = detect_file_lines_list[line_number + 2]
                                            #print(f"info_line.split(): {info_line.split()}")

                                            num_segments = int(info_line.split()[2])
                                            #print(f"num_segments: {num_segments}")

                                            #detects if number of meteors in file > 1. Currently don't know what to do if it is
                                            if (num_meteors := int(info_line.split()[1])) > 1:
                                                #adds file name back to list as file name is repeatedly appearing in FTPdetectinfo file but it then gets removed each time
                                                files_not_analysed.append(line[:-1])
                                                #log the anamoly of >1 meteor detection in an image as currently this script can't handle it
                                                logger.log(home_dir + "/log.csv", log_priority = "Low", log_type = "num.meteor detections > 1", logger_call_no = 2, details = f"FTPdetectinfo file in dir: {cwd} has a FITS file with multiple detections in it")

                                                # with open(home_dir + "_script_logfile.txt", "w") as logfile:
                                                #     logfile.write(f"{detect_file_lines_list[line_number][:-1]} has {num_meteors} meteor detections, please check images")
                                                #     print(f"{detect_file_lines_list[line_number][:-1]} has {num_meteors} meteor detections, please check images")

                                            files_not_analysed.remove(line[:-1])
                                            # print(f"files_not_analysed new: {files_not_analysed}")

                                            #opens the fits file which was found previously
                                            #[:-1] index removes the "\n" character from the line
                                            with fits.open(line[:-1]) as meteor_image:

                                                #get maxpixel image from meteor_image file
                                                maxpixel_image = meteor_image[1].data



                                                #get size of the maxpixel_image
                                                row_size = maxpixel_image.shape[0]
                                                col_size = maxpixel_image.shape[1]

                                                # print(f"row_size: {row_size}, col_size: {col_size}")

                                                # go through each line that has the position of the detection and create two list, one for the columns where the detection occurs and the other for the row
                                                detection_col = []
                                                detection_row = []

                                                for segment in range(num_segments):
                                                    detection_col.append(float(detect_file_lines_list[line_number + 3 + segment].split()[1]))
                                                    detection_row.append(float(detect_file_lines_list[line_number + 3 + segment].split()[2]))

                                                detection_pixels = [detection_col, detection_row]

                                                #create bounding box around meteor that is pixel larger than the limits of the detection
                                                # print(f"detection_box[0][0]: {detection_box[0][0]}")

                                                def bounding_box(box_coords, image_size, clearance):
                                                    bounding_box_left = math.floor(min(box_coords[0]) - clearance)
                                                    if bounding_box_left < 0:
                                                        bounding_box_left = 0
                                                    bounding_box_right = math.ceil(max(box_coords[0]) + clearance)
                                                    if bounding_box_right > image_size[1]:
                                                        bounding_box_right = image_size[1]

                                                    bounding_box_top = math.floor(min(box_coords[1]) - clearance)
                                                    if bounding_box_top < 0:
                                                        bounding_box_top = 0
                                                    bounding_box_bottom = math.ceil(max(box_coords[1]) + clearance)
                                                    if bounding_box_bottom > image_size[0]:
                                                        bounding_box_bottom = image_size[0]

                                                    box_bounds = {"left" : bounding_box_left,
                                                                    "right" : bounding_box_right,
                                                                    "bottom" : bounding_box_bottom,
                                                                    "top" : bounding_box_top}

                                                    return box_bounds


                                                #creates cropped image of the meteor
                                                crop_box_bounds = bounding_box(detection_pixels, maxpixel_image.shape, 10)
                                                crop_image = maxpixel_image[crop_box_bounds['top']:crop_box_bounds['bottom'], crop_box_bounds['left']:crop_box_bounds['right']]

                                                bigger_crop_box_bounds = bounding_box(detection_pixels, maxpixel_image.shape, 20)
                                                bigger_crop_image = maxpixel_image[bigger_crop_box_bounds['top']:bigger_crop_box_bounds['bottom'], bigger_crop_box_bounds['left']:bigger_crop_box_bounds['right']]


                                                #create plot of meteor_image and crop_image
                                                # has full image displayed on left side and then the two cropped images displayed on the right side
                                                # fig, axd = plt.subplot_mosaic([['big_image', 'big_image', 'small_image'],
                                                #                                 ['big_image', 'big_image', 'small_image_2']],
                                                #                                 )
                                                # #
                                                # axd['big_image'].imshow(maxpixel_image)
                                                # axd['small_image'].imshow(crop_image)
                                                # axd['small_image_2'].imshow(bigger_crop_image)
                                                #
                                                # # Create a Rectangle patch Rectangle((left column, top row), column width, row height, linewidth, edgecolor, facecolor)
                                                # rect = patches.Rectangle((crop_box_bounds['left'], crop_box_bounds['top']), (crop_box_bounds['right'] - crop_box_bounds['left']), (crop_box_bounds['bottom'] - crop_box_bounds['top']), linewidth=1, edgecolor='r', facecolor='none')
                                                #
                                                # # Add the patch to the big image
                                                # axd['big_image'].add_patch(rect)
                                                #
                                                # #change the size of the figure
                                                # fig.set_size_inches(18.5, 10.5)
                                                #
                                                # # display the plot
                                                # plt.show()
                                                #
                                                # # print(f"bigger_crop_image.shape: {bigger_crop_image.shape}")



                                                #save the Numpy array as a png using PIL
                                                im = Image.fromarray(bigger_crop_image)
                                                im.save(home_dir + "/" + dir_name + "_png" + "/" + line[:-6] + "_" + str(num_meteors) + ".png")



                                                #logging files that have been analysed
                                                logger.log(home_dir + "/log.csv", log_priority = "Low", log_type = "File processed", logger_call_no = 3, details = f"processed file:{line[:-1]}  in dir :{cwd}")

                            for item in files_not_analysed:
                                logger.log(home_dir + "/log.csv", log_priority = "Medium", log_type = "File not analysed", logger_call_no = 4, details = f"this file: {item} in dir: {cwd} was not analysed")
                            for item in fits_dont_exist:
                                logger.log(home_dir + "/log.csv", log_priority = "Medium", log_type = "FITS File not found", logger_call_no = 5, details = f"failed to find FITS file: {item} in dir: {cwd}")

                    except Exception as error_3:
                        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 9, details = f"Error: {error_3} - occured in dir: {cwd}")
                        print(f"Error occured: {error_3}")

            except Exception as error_2:
                logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 8, details = f"Error: {error_2} - occured in dir: {cwd}")
                print(f"Error occured: {error_2}")

    except Exception as error_1:
        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 7, details = f"Error: {error_1} - occured in dir: {cwd}")
        print(f"Error occured: {error_1}")

print("Cropping Script Finished")
logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Script finished", logger_call_no = 6, details = f"Script has finished")
