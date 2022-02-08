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
from pathlib import Path
from time import sleep, perf_counter


path_root = Path(__file__).parents[1]
print(f"path_root: {path_root}/RMS")
sys.path.insert(1, str(path_root) + "/RMS")

#sys path needs to be changed depending on machine or just have RMS added properly as a package
# sys.path.insert(1, "/home/fiachra/atom_projects/meteorml/RMS")

from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo



#file paths handled by input from user

#to get current working directory
#home_dir = os.getcwd()

#log command: (custom logger, just wanted something simple that I could look back on afterwards, could easily be changed out or removed)
#log(log_file_name="log.csv", log_priority, log_type, logger_call_no, details, log_time=datetime.datetime.now()):
print("need to change script to choose image type you want produced, detect only frames, maxframe or maxframe time sliced?")
cwd = input("directory path which has ConfirmedFiles and RejectedFiles folders: ")  #Meteor_Files directory path e.g. ~/Downloads/Meteor_Files
time_start = perf_counter()
os.chdir(cwd)

output_dir = input("output directory name for the pngs e.g. 20210104_pngs : ")

home_dir = os.getcwd()  #Meteor_Files directory
print(home_dir)

#gets a list of all the files available and then later as each file is analysed it will be removed from the list. At the end of the script any .fits files left in the list
fits_not_analysed = blackfill.search_dirs("", chosen_dir=home_dir, file_or_folder="file", search_subdirs=True)

top_directories = os.listdir()  #should be ConfirmedFiles and RejectedFiles folders amd Empty directories too
print(f"top_directories: {top_directories}")

#maybe add line for input to confirm that user has inputted the correct directory, otherwise get them to cancel the script

#creating new directories for the png versions of ConfirmedFiles and RejectedFiles
if output_dir not in top_directories:
    try:
        os.mkdir(output_dir)
    except:
        pass
    # may cause erros that it doesn't check if ConfirmedFiles_png or RejectedFiles_png directories exist
    try:
        os.mkdir(output_dir + "/" + "ConfirmedFiles_png")
    except:
        pass
    try:
        os.mkdir(output_dir + "/" + "RejectedFiles_png")
    except:
        pass


fits_dont_exist = []
# fits_not_analysed = []
#go through the folders in the top_directories and then only go through the ConfirmedFiles and RejectedFiles folders
for dir_name in top_directories:
    os.chdir(home_dir)
    top_dirs = os.getcwd()

    try:
        if dir_name == "ConfirmedFiles" or dir_name == "RejectedFiles" and os.path.isdir(dir_name) == True:    #if not directory containing empty directories

            os.chdir(home_dir +"/" + dir_name)    #now in either ConfirmedFiles or RejectedFiles
            Con_or_Rej_dir = os.getcwd()
            print("cwd1=", Con_or_Rej_dir)
            detection_folder_list = os.listdir() #should be BE0001....5, BE0001....6, etc.

            try:
                for night_dir in detection_folder_list:   #night_dir is a folder which is a single night of detections from an operating station

                    try:
                        print("\n")
                        # print("cwd2", cwd)
                        os.chdir(home_dir + "/" + dir_name + "/" + night_dir)
                        cwd = os.getcwd()
                        print("cwd3", cwd)

                        files_list = os.listdir()   #individual files in the detection folder, e.g. BE0001....fits, FTPdetectinfo....txt etc.

                        #list of fits files that are not anlaysed as they dont appear in the FTPdetectinfo file
                        # all files are added to the fits_not_analysed list so that they can be removed as they are analysed
                        # for element in files_list:
                        #     fits_not_analysed.append((element, cwd)) #this is the fastest method to make an actual copy, other methods exist for different scenarios when the list objects aren't strings
                        #elements from list are removed later



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
                                # if an FTPdetectinfo file name in the folder matches the naming scheme (2 letters at the start, nothing like pre-confiramtion appended to the end) then it will automatically choose that FTPdetectinfo_ file
                                if file == "FTPdetectinfo_" + night_dir + ".txt":
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
                                meteor_num = detection_entry[2]

                                if len((find_fits_file :=  blackfill.search_dirs(fits_file_name, chosen_dir=cwd, file_or_folder="file", exact_match=True, search_subdirs=False))) == 1:
                                    # print("test2")
                                    square_crop_image = blackfill.crop_detections_maxframe(detection_entry, cwd, time_slice=True)
                                    try:
                                        fits_not_analysed.remove((cwd, fits_file_name))
                                        #put in try statement as, if the same fits is analysed multiple times because of multiple detections in the same image
                                    except:
                                        pass
                                    #save the Numpy array as a png using PIL
                                    im = Image.fromarray(square_crop_image)
                                    im = im.convert("L")    #converts to grescale
                                    im.save(home_dir + "/" + output_dir + "/" + dir_name + "_png" + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")

                                elif len((find_fits_file := blackfill.search_dirs(fits_file_name, chosen_dir=home_dir, file_or_folder="file", exact_match=True, search_subdirs=True))) == 1:
                                    square_crop_image = blackfill.crop_detections_maxframe(detection_entry, find_fits_file[0][0], time_slice=True)
                                    try:
                                        fits_not_analysed.remove((find_fits_file[0][0], find_fits_file[0][1]))
                                        #put in try statement as, if the same fits is analysed multiple times because of multiple detections in the same image
                                    except:
                                        pass
                                    # save the Numpy array as a png using PIL
                                    im = Image.fromarray(square_crop_image)
                                    im = im.convert("L")    #converts to grescale
                                    im.save(home_dir + "/" + output_dir + "/" + dir_name + "_png" + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")

                                else:
                                    fits_dont_exist.append((fits_file_name, cwd))

                                #don't think is necessary and will only fill up the log file, would be necessary if script doesn't finish and needs to be run again but would need to add additional code to have the script pick up where it had left off
                                #logging files that have been analysed
                                # logger.log(home_dir + "/log.csv", log_priority = "Low", log_type = "File processed", logger_call_no = 3, details = f"processed file:{fits_file_name}  in dir :{cwd}")
                            #if the file isn't found in the cwd then search for the file in other directories
                                #logging any unanticipated errors that may occur
                                # except Exception as error_5:
                                #     logger.log(home_dir + "/log.csv", log_priority="High", log_type="file analysis failure", logger_call_no=10, details= traceback.format_exc())
                                #     print(traceback.format_exc())


                    except Exception as error_3:
                        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 9, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                        print(f"error_3: {traceback.format_exc()}")

            except Exception as error_2:
                logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 8, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                print(f"error_2: {traceback.format_exc()}")

    except Exception as error_1:
        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 7, details = f"Error: {error_1} - occured in dir: {cwd}")
        print(f"error_1: {traceback.format_exc()}")
for item in fits_not_analysed:
    if item[1][-4:] == "fits":

        logger.log(home_dir + "/log.csv", log_priority = "Medium", log_type = "File not analysed", logger_call_no = 4, details = f"this file: {item[1]} in dir: {item[0]} was not analysed")

for item in fits_dont_exist:
    logger.log(home_dir + "/log.csv", log_priority = "Medium", log_type = "FITS File not found", logger_call_no = 5, details = f"failed to find FITS file: {item[0]} in dir: {item[1]}")

print("Cropping Script Finished")
logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Script finished", logger_call_no = 6, details = f"Script has finished")

time_end = perf_counter()
print(f"time: {time_end - time_start}")
