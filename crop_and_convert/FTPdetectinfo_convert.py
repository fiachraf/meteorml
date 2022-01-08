import os
import sys
import fiachra_python_logger as logger
import traceback
import time

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

                        #as some folders have multiple FTPdetectinfo files, need to sort through them and pick correct one
                        FTPdetect_list = []
                        for item in files_list:
                            #search through files for FTPdetectinfo file, might be multiple files or file named differently
                            if item.find("FTPdetectinfo") != -1:
                                FTPdetect_list.append(item)

                        if len(FTPdetect_list) == 0:
                            print("no FTPdetectinfo file found")
                            time.sleep(2)
                            logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "FTPdetectinfo File not found", logger_call_no = 1, details = f"failed to find FTPdetectinfo file in dir: {cwd}")

                        else:

                            chosen_FTP_file = ""

                            print("\n")
                            #used 2 seperate for loops so that it searches for the correct file first and if that fails then it searches for a file with the CAMS station code
                            for index_1, file_1 in enumerate(FTPdetect_list):
                                # print(f"file[:-4]: {file[:-4]}")
                                # if an FTPdetectinfo file name in the folder matches the naming scheme (2 letters at the start, nothing like pre-confiramtion appended to the end) then it will automatically choose that FTPdetectinfo_ file
                                print(f"file_1[-26:]: {file_1[-26:]}, folders[7:]: {folders[7:]}")
                                if file_1 == "FTPdetectinfo_" + folders + ".txt":
                                    print(f"correct form FTPdetectinfo file present in dir: {cwd}")
                                    chosen_FTP_file = "done"
                                    break

                            #if it doesn't find the correctly named file it will loop through the file list again to try find the incorrectly named file that has the data in it
                            if chosen_FTP_file == "":
                                for index_2, file_2 in enumerate(FTPdetect_list):
                                    #detect incorrectly labelled but fully completed FTPdetectinfo files
                                    #file[:-26] is everything after the incorrect station code
                                    if file_2[-26:] == folders[7:] + ".txt":

                                        #read all the lines of the file
                                        with open(file_2, "r") as read_FTP_file:
                                            read_FTP_file_lines = read_FTP_file.readlines()
                                            for read_FTP_index, read_FTP_line in enumerate(read_FTP_file_lines):
                                                if read_FTP_line.find(".fits") != -1:
                                                    #change the line that has the fits file name to have the correct fits file name
                                                    old_station_code_start_index = 3
                                                    old_station_code_end_index = read_FTP_line[3:].find("_") + 3
                                                    # print(f"old_station_code_end_index: {old_station_code_end_index}")
                                                    new_FTP_line = read_FTP_line[:3] + folders[:6] + read_FTP_line[old_station_code_end_index:]
                                                    read_FTP_file_lines[read_FTP_index] = new_FTP_line

                                                    #change the line 2 lines down that has an incorrect camera code
                                                    #camera code = folders[:6] i.e. BE0001
                                                    old_other_line = read_FTP_file_lines[read_FTP_index + 2]
                                                    new_other_line = ""
                                                    new_other_line = new_other_line + folders[:6]
                                                    new_other_line = new_other_line + old_other_line[old_other_line.find(" "):]
                                                    read_FTP_file_lines[read_FTP_index + 2] = new_other_line

                                            #write the new file line by line
                                            with open("FTPdetectinfo_" + folders + ".txt", "w") as write_FTP_file:
                                                for write_FTP_file_lines in read_FTP_file_lines:
                                                    write_FTP_file.write(write_FTP_file_lines)

                                        chosen_FTP_file = "done"
                                        break

                                        #could've used code like below, however that erased all the calibration info as I don't have thr proper files or haven't done things properly. Code has been left here to maybe help someone else implement it in future. Would need to do the loops differently

                                        # #read data from FTPdetectinfo file
                                        # FTP_file = FTPdetectinfo.readFTPdetectinfo(cwd, file, ret_input_format=True)
                                        # #returns tuple of data that can be written to new file using FTPdetectinfo.writeFTPdetectinfo(meteor_list, ff_directory, file_name, cal_directory, cam_code, fps, calibration=None)
                                        # #FTP_file[0] is the cam_code
                                        # #FTP_file[1] is the fps
                                        # #FTP_file[2] is the meteor_list
                                        #
                                        # # new_FTP_file = FTP_file[:]  #can't edit contents of tuple so variables and list need to be made and then tuple can be made after the for loop
                                        #
                                        # new_cam_code = folders[:6]   #cam_code should be the same as the first 6 characters of the folder it is in
                                        # print(f"new_cam_code: {new_cam_code}")
                                        # new_fps = FTP_file[1]   #fps should be the same
                                        # print(f"new_fps: {new_fps}")
                                        # old_meteor_list = FTP_file[2]
                                        # print(f"old_meteor_list: {old_meteor_list}")
                                        # new_meteor_list = []
                                        # #loop through each image entry in the FTPdetectinfo file and analyse each image
                                        #
                                        # for detection_index, detection_entry in enumerate(old_meteor_list):
                                        #     new_detection_entry = detection_entry[:]
                                        #     print(f"new_detection_entry: {new_detection_entry}")
                                        #     new_fits_file_name = f"FF_{folders[:6]}_{detection_entry[0][-32:]}"
                                        #     print(f"new_fits_file_name: {new_fits_file_name}")
                                        #     new_detection_entry[0] = new_fits_file_name
                                        #     new_meteor_list.append(new_detection_entry)
                                        # new_FTP_file = (new_cam_code, new_fps, new_meteor_list)
                                        # print(f"new_FTP_file: {new_FTP_file}")
                                        #
                                        # FTPdetectinfo.writeFTPdetectinfo(new_meteor_list, cwd, "FTPdetectinfo_" + folders + ".txt", cwd, new_cam_code, new_fps)

                                        # chosen_FTP_file = "done"
                                        # break




                            #if no FTPdetectinfo auto chosen then, print options available and have the user choose the correct one
                            if chosen_FTP_file == "":
                                print("no correct FTPdetectinfo file found")
                                # print("\n choose best FTPdetectinfo file")
                                # for possible_file_index, possible_file in enumerate(FTPdetect_list):
                                #     print(f"file_name: {possible_file}, index: {possible_file_index}")
                                #
                                # #take first input and if bad input (empty, not a number, out of range) is inputted then prompt user again
                                # while True:
                                #     try:
                                #         chosen_FTP_file_index = int(input("please enter index of FTPdetectinfo file to be used to pick out meteors from the image: "))
                                #         if chosen_FTP_file_index >= 0 and chosen_FTP_file_index < len(FTPdetect_list):
                                #             chosen_FTP_file = FTPdetect_list[chosen_FTP_file_index]
                                #             break
                                #
                                #     except Exception as error_4:
                                #         print("non number value entered, try again")




                    except Exception as error_3:
                        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 9, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                        print(f"error_3: {traceback.format_exc()}")

            except Exception as error_2:
                logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 8, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                print(f"error_2: {traceback.format_exc()}")

    except Exception as error_1:
        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 7, details = f"Error: {error_1} - occured in dir: {cwd}")
        print(f"error_1: {traceback.format_exc()}")

print("Script Finished")
logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Script finished", logger_call_no = 6, details = f"Script has finished")
