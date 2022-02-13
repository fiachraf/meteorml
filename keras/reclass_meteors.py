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


     def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self.pngdir = png_dir
        self.png_path = file_path
        #search_dirs(search_term, chosen_dir, file_or_folder="", exact_match=False, search_subdirs=False)
        #file_name[3:18] is the directory that it came from
        print("test1", self.name[3:37])
        print("test2", search_dirs(self.name[3:37], Confirmed_FITS_dir))
        self.con_fits_dir = search_dirs(self.name[3:37], Confirmed_FITS_dir)[0]
        self.rej_fits_dir = search_dirs(self.name[3:37], Rejected_FITS_dir)[0]
        #might need to get basename for self.con_fits_dir in search term
        print("test3", self.con_fits_dir)
        con_FTP_file = search_dirs("FTPdetectinfo_" + os.path.basename(self.con_fits_dir) + ".txt" , self.con_fits_dir)
        rej_FTP_file = search_dirs("FTPdetectinfo_" + os.path.basename(self.rej_fits_dir) + ".txt" , self.rej_fits_dir)
        self.con_FTP = os.path.join(con_FTP_file[0], con_FTP_file[1])
        self.rej_FTP = os.path.join(rej_FTP_file[0], rej_FTP_file[1])

def con_butt(event):
    print("test conf butt")
def rej_butt(event):
    print("test rej butt")




def plot_meteor_img(meteor_img):
    fig, ax1  = plt.subplots()
    fig.set_size_inches(6,6)
    img = mpimg.imread(meteor_img.png_path)
    print("test4", meteor_img.name)
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






#TODO: change to read proper file and have proper indices
with open(initial_dir + "/" + csv_file, mode="r") as csv_file:
    csv_reader = csv.reader(csv_file)
    #row_index = 0
    for row_index, row in enumerate(csv_reader):
        if row_index == 0:
            continue
        #row[0] is the label, row[1] is the prediction, row[2] is True Prediction, row[3] is the file path of one of the dupes
        image_1 = meteor_image(row)
        print(f"image_1.name: {image_1.name}\n\
                image_1.png_dir: {image_1.pngdir}\n\
                image_1.con_fits_dir: {image_1.con_fits_dir}\n\
                image_1.rej_fits_dir: {image_1.rej_fits_dir}\n\
                image_1.con_FTP: {image_1.con_FTP}\n\
                image_1.rej_FTP: {image_1.rej_FTP}")

        plot_meteor_img(image_1)


