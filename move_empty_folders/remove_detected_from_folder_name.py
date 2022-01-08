import os
import sys

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

                    if folders[-9:] == "_detected":
                        try:
                            os.rename(folders, folders[:-9])


                        except Exception as error_3:
                            logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Folder rename", logger_call_no = 9, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                            print(f"error_3: {traceback.format_exc()}")

            except Exception as error_2:
                logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 8, details = f"Error: {traceback.format_exc()} - occured in dir: {cwd}")
                print(f"error_2: {traceback.format_exc()}")

    except Exception as error_1:
        logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Unknown error", logger_call_no = 7, details = f"Error: {error_1} - occured in dir: {cwd}")
        print(f"error_1: {traceback.format_exc()}")

print("Cropping Script Finished")
logger.log(home_dir + "/log.csv", log_priority = "High", log_type = "Script finished", logger_call_no = 6, details = f"Script has finished")
