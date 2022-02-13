import os
import numpy as np
import csv
import matlpotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from pathlib import Path


path_root = Path(__file__).parents[1]
#print(f"path_root: {path_root}/RMS")
sys.path.insert(1, str(path_root) + "/RMS")
sys.path.insert(1, str(path_root) + "crop_and_convert")

from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
from crop_and_convert import blackfill

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
csv_file = input("will need to change script slightly to accomodate for different csv files\n
        csv_file containing file names: ")
Confirmed_FITS_dir = os.path.dirname(png_dir) + "/" + "ConfirmedFiles"
Rejected_FITS_dir = os.path.dirname(png_dir) + "/" + "RejectedFiles"




class meteor_image:


     def __init__(self, file_name):
        self.name = file_name
        self.png_dir = png_dir
        #search_dirs(search_term, chosen_dir, file_or_folder="", exact_match=False, search_subdirs=False)
        #file_name[3:18] is the directory that it came from
        self.con_fits_dir = blackfill.search_dirs(file_name[3:18]), Confirmed_FITS_dir, file_or_folder="folder")
        self.rej_fits_dir = blackfill.search_dirs(file_name[3:18]), Rejected_FITS_dir, file_or_folder="folder")
        #might need to get basename for self.con_fits_dir in search term
        self.con_FTP = blackfill.search_dirs("FTPdetectinfo_" + self.con_fits_dir + ".txt" , self.con_fits_dir, file_or_folder="file")
        self.rej_FTP = blackfill.search_dirs("FTPdetectinfo_" + self.rej_fits_dir + ".txt" , self.rej_fits_dir, file_or_folder="file")


#TODO: change to read proper file and have proper indices
with open(csv_file, mode="r") as csv_file:
    csv_reader = csv.reader(csv_file)
    row_index = 0
    for row in csv_reader:
        #row[0] is the file name, not the full path, row[1] is the number of times it occurs
        image_1 = meteor_image(row[0])
        print(f"image_1.name: {image_1.name}\n
                image_1.png_dir: {image_1.png_dir}\n
                image_1.con_fits_dir: {image_1.con_fits_dir}\n
                image_1.rej_fits_dir: {image_1.rej_fits_dir}\n
                image_1.con_FTP: {image_1.con_FTP}\n
                image_1.rej_FTP: {image_1.rej_FTP}")
