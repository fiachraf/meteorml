import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from pathlib import Path


path_root = Path(__file__).parents[1]
#print(f"path_root: {path_root}/RMS")
sys.path.insert(1, str(path_root) + "/RMS")
sys.path.insert(1, str(path_root) + "/crop_and_convert")

from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
import blackfill

#data to plot
# x1 = np.array()
# y1 = np.array()
#
# fig = plt.figure
# ax = fig.subplots()
# plt.subplots_adjust(lefft = 0.3, bottom = 0.25)
# p, = ax.plot(x1, y1, color="blue", marker="o")
#
#
# #defining function to add line plot
# def add(val):
#     x2 = np.array()
#     y2 = np.array()
#     ax.plot(x2, y2, color="green", marker="o")

#defining button to add and its functionality

png_dir = input("directory with folders 1 and 0 which contain png images to view through: ")
csv_file = input("will need to change script slightly to accomodate for different csv files\ncsv_file containing file names: ")
Confirmed_FITS_dir = os.path.dirname(png_dir) + "/" + "ConfirmedFiles"
Rejected_FITS_dir = os.path.dirname(png_dir) + "/" + "RejectedFiles"

print(f"Confirmed_FITS_dir: {Confirmed_FITS_dir}")
initial_dir = os.getcwd()
os.chdir(png_dir)
cwd1 = os.getcwd()

def search_dirs(search_term, top_search_dir):
    #file_list = os.walk(top_search_dir)
    #for fits_file in file_list:
    #    if fits_file.find(search_term) != -1:
    #        return fits_file

    for root, dirs, files in os.walk(top_search_dir):
        for file_name in files:
            if file_name.find(search_term) != -1:
                return (root, file_name)
    return ("","")


class meteor_image:


     def __init__(self, label, prediction, True_prediction, file_path):
        self.name = os.path.basename(file_path)
        self.pngdir = png_dir
        self.png_path = file_path
        #search_dirs(search_term, chosen_dir, file_or_folder="", exact_match=False, search_subdirs=False)
        #file_name[3:18] is the directory that it came from
        #print("test1", self.name[3:37])
        #print("test2", search_dirs(self.name[3:37], Confirmed_FITS_dir))
        self.con_fits_dir = search_dirs(self.name[3:37], Confirmed_FITS_dir)[0]
        self.rej_fits_dir = search_dirs(self.name[3:37], Rejected_FITS_dir)[0]
        #might need to get basename for self.con_fits_dir in search term
        #print("test3", self.con_fits_dir)
        if len(self.con_fits_dir) < 3 and len(self.rej_fits_dir) < 3:
            print("MAjor Error")
        if len(self.con_fits_dir) < 3:
            self.con_fits_dir = os.path.join(Confirmed_FITS_dir, os.path.basename(self.rej_fits_dir))
        if len(self.rej_fits_dir) < 3:
            self.rej_fits_dir = os.path.join(Rejected_FITS_dir, os.path.basename(self.con_fits_dir))

        con_FTP_file = search_dirs("FTPdetectinfo_" + os.path.basename(self.con_fits_dir) + ".txt" , self.con_fits_dir)
        rej_FTP_file = search_dirs("FTPdetectinfo_" + os.path.basename(self.rej_fits_dir) + ".txt" , self.rej_fits_dir)
        self.con_FTP = os.path.join(con_FTP_file[0], con_FTP_file[1])
        self.rej_FTP = os.path.join(rej_FTP_file[0], rej_FTP_file[1])

        #print("test1", os.path.join(self.con_fits_dir, self.name[:37] + ".fits"))
        if os.path.isfile(os.path.join(self.con_fits_dir, self.name[:37] + ".fits")) == True and  os.path.isfile(os.path.join(self.rej_fits_dir, self.name[:37] + ".fits")) == True:
            self.con_or_rej = "both"
            self.fits_file = os.path.join(self.con_fits_dir, self.name[:37] + ".fits")
        elif os.path.isfile(os.path.join(self.con_fits_dir, self.name[:37] + ".fits")) == True:
            self.con_or_rej = "con"
            self.fits_file = os.path.join(self.con_fits_dir, self.name[:37] + ".fits")
        elif os.path.isfile(os.path.join(self.rej_fits_dir, self.name[:37] + ".fits")) == True:
            self.con_or_rej = "rej"
            self.fits_file = os.path.join(self.rej_fits_dir, self.name[:37] + ".fits")
        else:
            self.con_or_rej = None
        self.meteor_num = int(self.name[-5])

def del_FTP_entries(meteor_image_2):
    """
    deletes entries in FTP files that shouldn;t be there. i.e. confirmed meteor detections should not be in the FTP file that is in the rejected folder and vice versa
    input is a meteor_image class object as that will contain a lot of the necessary info
    """
    #returns None to deliberately cause problems later on, TODO manage return None
    if meteor_image_2.con_or_rej == "both" or meteor_image_2.con_or_rej == None:
        return None
    #deleting entry from wrong FTP file so need opposite FTP file from classification
    elif meteor_image_2.con_or_rej == "con":
        chosen_FTP_file = meteor_image_2.rej_FTP
    elif meteor_image_2.con_or_rej == "rej":
        chosen_FTP_file = meteor_image_2.con_FTP

    temp_FTP_file = chosen_FTP_file + ".tmp"
    with open(chosen_FTP_file, "r") as detect_file:

        detect_file_lines_list = detect_file.readlines()
        write_file_lines_list = detect_file_lines_list[:]
        # detect_file.seek(0) #resets pointer to start of file
        #put this following line just before the write statement so that if things go wrong it doesn't wipe the file
        #detect_file.truncate() #deletes all lines after this point as they will all be wrt
        #or create temporary file with modified data, and then rename temp file to original file name and replace original file
        #dupes are only created in directories that have the same FTP files
        #this code only executes if the fits appears in only one directory and so all mentions of the .fits file in the FTP in the wrong directory should be deleted
        #if the .fits file appears in both directories this code won't execute and the FTP file will be left alone
        #should also delete multiple instances of the same .fits being mentioned
        for line_number, line in enumerate(detect_file_lines_list):
            if line.find(os.path.basename(meteor_image_2.fits_file)) != -1:
                #gets the num frames the meteor is in and thus the number of lines in the entry
                #need to delete the dashed line above the file name, line with the file name, line with calibration inoformation, line with cam details, and lines with detection details
                detection_start_line = line_number - 1

                #make sure it is the correct detection
                detection_no = int(detect_file_lines_list[line_number + 2][7:11]
                if detection_no != meteor_image_2.meteor_num:
                    continue


                #detect_file_lines_list[line_number+2][12:16] gets the number of frame lines
                num_lines_to_del = int(detect_file_lines_list[line_number + 2][12:16]) + 4
                #use del list[start:end] to remove elements including start, up to end
                del write_file_lines_list[detection_start_line : detection_start_line + num_lines_to_del]
        #print(f"temp_FTP_file: {temp_FTP_file}")
        with open(temp_FTP_file, "w") as temp_file:
            temp_file.writelines(write_file_lines_list)
    #os.replace(temp_FTP_file, chosen_FTP_file)
        #TODO test deleting function





#only applies for dupe images as these dupe images have the same FTP file on both Confirmed and Rejected folders and only have the FITS file in the correct directory
#some dupes have both the same FTP file and same FITS files in both directories as there were at least one Confirmed and one Rejected in the same FITS file
#TODO: Add functions to remove entries from FTP files if FITS file not in directory
#TODO: Add unsure button, add functionality to buttons

def plot_meteor_img(meteor_img):
    fig, ax1  = plt.subplots()
    fig.set_size_inches(6,6)
    img = mpimg.imread(meteor_img.png_path)
    #print("test4", meteor_img.name)
    meteor_img_name = meteor_img.name
    ax1.imshow(img)
    ax1.set_title(meteor_img.name)
    #plt.xlabel(f"Label: {os.path.dirname(meteor_img.png_path)[-1], Prediction
    axconf = fig.add_axes([0.55,0,0.1,0.1])
    axrej = fig.add_axes([0.35,0,0.1,0.1])
    bconf = Button(axconf, "Confirmed")
    brej = Button(axrej, "Rejected")
    bconf.on_clicked(con_butt)
    brej.on_clicked(rej_butt)


    plt.show()



#if the fits file exists in both directories and detection needs to be manually checked, it is added to to both_img_list else it is put in the one_img_list and will be automatically dealt with
both_img_list = []
one_img_list = []
#if for some reason an image doesn't fall into either of the catgories it will be appended to the problem_list
problem_list = []

with open(initial_dir + "/" + csv_file, mode="r") as csv_file:
    csv_reader = csv.reader(csv_file)
    #row_index = 0
    for row_index, row in enumerate(csv_reader):
        if row_index == 0:
            continue
        #row[0] is the label, row[1] is the prediction, row[2] is True Prediction, row[3] is the file path of one of the dupes
        image_1 = meteor_image(row[0], row[1], row[2], row[3])
        if image_1.con_or_rej == "both":
            both_img_list.append(image_1)
        elif image_1.con_or_rej == "con" or image_1.con_or_rej == "rej":
            one_img_list.append(image_1)
        else:
            problem_list.append(image_1)

        #del_FTP_entries(image_1)

        #plot_meteor_img(image_1)


for image_2 in one_img_list:
#    print(f"image_1.name: {image_1.name}\n\
#            image_1.png_dir: {image_1.pngdir}\n\
#            image_1.png_path: {image_1.png_path}\n\
#            image_1.con_fits_dir: {image_1.con_fits_dir}\n\
#            image_1.rej_fits_dir: {image_1.rej_fits_dir}\n\
#            image_1.con_FTP: {image_1.con_FTP}\n\
#            image_1.rej_FTP: {image_1.rej_FTP}\n\
#            image_1.con_or_rej: {image_1.con_or_rej}\n\
#            image_1.fits_file: {image_1.fits_file}")
    try:
        del_FTP_entries(image_2)

        pause_input = input("waiting for input, check if .tmp is done correctly then remove commented line 148 and restart script so that it will then replace the FTP files, if files not done correctly just press Ctrl + C")
    except Exception as error_1:
        print(f"Error: {error_1}")

        print(f"image_2.name: {image_2.name}\n\
                image_2.png_dir: {image_2.pngdir}\n\
                image_2.png_path: {image_2.png_path}\n\
                image_2.con_fits_dir: {image_2.con_fits_dir}\n\
                image_2.rej_fits_dir: {image_2.rej_fits_dir}\n\
                image_2.con_FTP: {image_2.con_FTP}\n\
                image_2.rej_FTP: {image_2.rej_FTP}\n\
                image_2.con_or_rej: {image_2.con_or_rej}\n\
                image_2.fits_file: {image_2.fits_file}\n\
                image_2.meteor_num: {image_2.meteor_num}")

print(f"len(both_img_list): {len(both_img_list)}")
print(f"len(problem_list): {len(problem_list)}")
with open(os.path.join(initial_dir, "need_reclassifying.csv"), "w") as reclass_csv:
    reclass_csv_writer = csv.writer(reclass_csv, delimiter=",")
    reclass_csv_writer.writerow(["name", "pngdir", "png_path", "con_fits_dir", "rej_fits_dir", "con_FTP", "rej_FTP", "con_or_rej", "fits_file", "meteor_num"])
    for image_3 in both_img_list:
        reclass_csv_writer.writerow([image_3.name, image_3.pngdir, image_3.png_path, image_3.con_fits_dir, image_3.rej_fits_dir, image_3.con_FTP, image_3.rej_FTP, image_3.con_or_rej, image_3.fits_file, image_3.meteor_num])
        #print(f"image_1.name: {image_1.name}\n\
        #        image_1.png_dir: {image_1.pngdir}\n\
        #        image_1.png_path: {image_1.png_path}\n\
        #        image_1.con_fits_dir: {image_1.con_fits_dir}\n\
        #        image_1.rej_fits_dir: {image_1.rej_fits_dir}\n\
        #        image_1.con_FTP: {image_1.con_FTP}\n\
        #        image_1.rej_FTP: {image_1.rej_FTP}\n\
        #        image_1.con_or_rej: {image_1.con_or_rej}\n\
        #        image_1.fits_file: {image_1.fits_file}")

if len(problem_list) > 0:
    with open(os.path.join(intial_dir, "problem_files.csv"), "w") as prob_csv:
        prob_csv_writer = csv.writer(prob_csv, delimiter=",")
        prob_csv_writer.writerow(["FITS file", "confirmed FTP File", "rejected FTP file"])
        for image_4 in problem_list:
            reclass_csv_writer.writerow([image_3.fits_file, image_3.con_FTP, image_3.rej_FTP])
    #print(f"image_1.name: {image_1.name}\n\
    #        image_1.png_dir: {image_1.pngdir}\n\
    #        image_1.png_path: {image_1.png_path}\n\
    #        image_1.con_fits_dir: {image_1.con_fits_dir}\n\
    #        image_1.rej_fits_dir: {image_1.rej_fits_dir}\n\
    #        image_1.con_FTP: {image_1.con_FTP}\n\
    #        image_1.rej_FTP: {image_1.rej_FTP}\n\
    #        image_1.con_or_rej: {image_1.con_or_rej}\n\
    #        image_1.fits_file: {image_1.fits_file}")
