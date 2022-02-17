import os
import sys
import shutil


#to get current working directory
#home_dir = os.getcwd()

#to change current working directory
# os.chdir()


#list contents of directory, if brackets empty, lists current directory contents, otherwise will list contents of directory at specified path
#os.listdir()

cwd = input("directory path which has ConfirmedFiles and RejectedFiles folders: ")
os.chdir(cwd)

home_dir = os.getcwd()
print(home_dir)

def move_empty_folders(Files_folder):

    top_directories = os.listdir()  #should be ConfirmedFiles and RejectedFiles folders

    for dir_name in top_directories:
        if dir_name == "ConfirmedFiles" or dir_name == "RejectedFiles":
            cwd = os.chdir(home_dir)

            #make folder for empty directories
            try:
                EmptyDirectories = Files_folder + "/Empty_" + dir_name
                os.mkdir(EmptyDirectories)
            except OSError as error:
                print(error)

            #change cwd to ConfirmedFiles / RejectedFiles directory
            os.chdir(dir_name)
            cwd = os.getcwd()
            Files_dirs_list = os.listdir()


            for folder in Files_dirs_list:
                if len(os.listdir(folder)) == 0:
                    shutil.move(folder, EmptyDirectories)


move_empty_folders(home_dir)
