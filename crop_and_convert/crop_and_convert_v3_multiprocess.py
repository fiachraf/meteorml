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
import multiprocessing
from functools import partial


path_root = Path(__file__).parents[1]
print(f"path_root: {path_root}/RMS")
sys.path.insert(1, str(path_root) + "/RMS")

#sys path needs to be changed depending on machine or just have RMS added properly as a package
# sys.path.insert(1, "/home/fiachra/atom_projects/meteorml/RMS")

from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo

def process_FTP_detections(FTP_file, FTP_dir, files_not_analysed_list):
    """
    processes all the meteor detections in FTP file
    FTP_dir is the directory where the FTP file is located
    """
    # print(f"FTP_dir: {FTP_dir}")
    # print(f"cwd11: {os.getcwd()}")

    # os.chdir(FTP_dir)
    #read data from FTPdetectinfo file
    FTP_file = FTPdetectinfo.readFTPdetectinfo(FTP_dir, FTP_file)
    #loop through each image entry in the FTPdetectinfo file and analyse each image
    for detection_entry in FTP_file:
        #for timing performance
        #cpu_time_start = perf_counter()

        fits_file_name = detection_entry[0]
        meteor_num = detection_entry[2]
        if len((find_fits_file :=  blackfill.search_dirs(fits_file_name, chosen_dir=FTP_dir, file_or_folder="file", exact_match=True, search_subdirs=False))) == 1:
            square_crop_image = blackfill.crop_detections(detection_entry, FTP_dir)
            # try:
            #     # print("test8")
            #     # print(f"FTP_dir : {FTP_dir}, fits_file_name: {fits_file_name}")
            #     # print(f"files_not_analysed_list[280:283] : {files_not_analysed_list[280:283]}")
            #     # print(f"pre len(files_not_analysed_list: {len(files_not_analysed_list)})")
            #     files_not_analysed_list.remove((FTP_dir, fits_file_name))
            #     # print(f"post len(files_not_analysed_list: {len(files_not_analysed_list)})")
            #     #put in try statement as, if the same fits is analysed multiple times because of multiple detections in the same image
            # except Exception as error_7:
            #     # print("fuck")
            #     # print((FTP_dir, fits_file_name))
            #     # print(files_not_analysed_list)
            #     # logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 10, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
            #     # print(f"error_2: {traceback.format_exc()}")
            #     pass

            #for timing CPU performance
            #cpu_time_end = perf_counter()
            #timing IO performance
            #io_time_start = perf_counter()

            #save the Numpy array as a png using PIL
            im = Image.fromarray(square_crop_image)
            im = im.convert("L")    #converts to grescale
            im.save(home_dir + "/" + dir_name + "_png" + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")
            #io_time_end = perf_counter()





        elif len((find_fits_file := blackfill.search_dirs(fits_file_name, chosen_dir=home_dir, file_or_folder="file", exact_match=True, search_subdirs=True))) == 1:
            square_crop_image = blackfill.crop_detections(detection_entry, find_fits_file[0][0])
            # try:
            #     # print("test7")
            #     files_not_analysed_list.remove((find_fits_file[0][0], find_fits_file[0][1]))
            #     #put in try statement as, if the same fits is analysed multiple times because of multiple detections in the same image
            # except:
            #     # print("fuck2")
            #     pass


            #save the Numpy array as a png using PIL
            im = Image.fromarray(square_crop_image)
            im = im.convert("L")    #converts to grescale
            im.save(home_dir + "/" + dir_name + "_png" + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")



        else:
            fits_dont_exist.append((fits_file_name, FTP_dir))
    return

        #don't think is necessary and will only fill up the log file, would be necessary if script doesn't finish and needs to be run again but would need to add additional code to have the script pick up where it had left off
        #logging files that have been analysed
        # logger.log(home_dir + "/log.csv", log_priority = "Low", log_type = "File processed", logger_call_no = 3, details = f"processed file:{fits_file_name}  in dir :{cwd}")
        #logging any unanticipated errors that may occur
        # except Exception as error_5:
        #     logger.log(home_dir + "/log.csv", log_priority="High", log_type="file analysis failure", logger_call_no=10, details= traceback.format_exc())
        #     print(traceback.format_exc())

def check_FTP_exists(FTP_dir):
    """
    checks if an FTPdetectinfo file exists in directory cwd5
    If file with correct station code doesn't exist then presents list of other FTPdetectinfo files that could be used instead
    MAY CAUSE PROBLEMS WITH MULTIPROCESSING
    Should change function to have it convert FTPdetectinfo file with incorrect station code but correct detections to file with correct station code and detections
    If no FTP file found then it prints the directory, logs the directory sleeps for 2 seconds and then continues on
    """

    files_list = os.listdir(FTP_dir)   #individual files in the detection folder, e.g. BE0001....fits, FTPdetectinfo....txt etc.

    #as some folders have multiple FTPdetectinfo files, need to sort through them and pick correct one
    FTPdetect_list = []
    for item in files_list:
        #search through files for FTPdetectinfo file, might be multiple files or file named differently
        if item.find("FTPdetectinfo") != -1:
            FTPdetect_list.append(item)
    if len(FTPdetect_list) == 0:
        print("no FTPdetectinfo file found")
        sleep(2)
        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "FTPdetectinfo File not found", logger_call_no = 1, details = f"failed to find FTPdetectinfo file in dir: {FTP_dir}")
        return None

    else:

        chosen_FTP_file = ""
        print("\n")
        for index, file in enumerate(FTPdetect_list):
            # if an FTPdetectinfo file name in the folder matches the naming scheme (2 letters at the start, nothing like pre-confiramtion appended to the end) then it will automatically choose that FTPdetectinfo_ file
            # print(f"file: {file}")
            # print(f"'FTPdetectinfo_' + FTP_dir + ''.txt': FTPdetectinfo_{os.path.basename(FTP_dir)}.txt")
            if file == "FTPdetectinfo_" + os.path.basename(FTP_dir) + ".txt":
                chosen_FTP_file = file
                return chosen_FTP_file

        #if no FTPdetectinfo auto chosen then, return None and log error
        if chosen_FTP_file == "":
            logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "FTPdetectinfo wrong station code", logger_call_no = 2, details = f"wrong station code for FTPdetectinfo file in dir: {FTP_dir}")
            return None

def process_folder(detection_folder, files_not_analysed_list):

    if (FTP_file_name  := check_FTP_exists(os.path.abspath(detection_folder))) == None:
        return None
    else:
        try:
            process_FTP_detections(FTP_file_name, os.path.abspath(detection_folder), files_not_analysed_list)
        except Exception as error_11:
            logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 11, details = f"Error: {error_11} - occured in dir: {cwd}")
            print(f"error_11: {traceback.format_exc()}")



if __name__ == "__main__":


    #file paths handled by input from user

    #log command: (custom logger, just wanted something simple that I could look back on afterwards to see which files exist or aren't analysed or cause problems, could easily be changed out or removed)
    #log command usage
    #log(log_file_name="log.csv", log_priority, log_type, logger_call_no, details, log_time=datetime.datetime.now()):

    cwd = input("directory path which has ConfirmedFiles and RejectedFiles folders: ")  #Meteor_Files directory path e.g. ~/Downloads/Meteor_Files
    time_start = perf_counter()
    os.chdir(cwd)

    home_dir = os.getcwd()  #Meteor_Files directory

    #gets a list of all the files available and then later as each file is analysed it will be removed from the list. At the end of the script any .fits files left in the list have their name written to the log file
    fits_not_analysed = blackfill.search_dirs("", chosen_dir=home_dir, file_or_folder="file", search_subdirs=True)
    #create a list of .fits that are mentioned in the FTPdetectinfo files but are not present in the dataset, written to log file at the end of the script
    fits_dont_exist = []

    top_directories = os.listdir()  #should be ConfirmedFiles and RejectedFiles folders amd Empty directories too
    #print(f"top_directories: {top_directories}")

    #maybe add line for input to confirm that user has inputted the correct directory, otherwise get them to cancel the script

    #creating new directories for the png versions of ConfirmedFiles and RejectedFiles if they don't already exist
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
        top_dirs = os.getcwd()

        try:
            if dir_name == "ConfirmedFiles" or dir_name == "RejectedFiles" and os.path.isdir(dir_name) == True:    #if not directory containing empty directories

                os.chdir(home_dir +"/" + dir_name)    #now in either ConfirmedFiles or RejectedFiles
                Con_or_Rej_dir = os.getcwd()
                detection_folder_list = os.listdir() #should be BE0001....5, BE0001....6, etc.


                try:
                    'insert multiprocessing part here to spawn off new process for every detection folder'
                    num_processes = 10 #max num processes to have going at once
                    with multiprocessing.Manager() as manager:
                        l = manager.list(fits_not_analysed)
                        # for i in range(20):
                        #     print(l[i])
                        with multiprocessing.Pool(processes = num_processes) as pool:
                            results = pool.map(partial(process_folder, files_not_analysed_list = l), detection_folder_list)
                        #dir_path might not be needed, the second argument for partial should be an argument for the function that does not change for each function call, e.g. directory where to save pngs maybe?
                    #for night_dir in detection_folder_list:   #night_dir is a folder which is a single night of detections from an operating station





                        #
                        # except Exception as error_3:
                        #     logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 9, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                        #     print(f"error_3: {traceback.format_exc()}")

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
