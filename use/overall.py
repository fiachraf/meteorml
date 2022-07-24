import png_gen
import make_pred
import sys
import os
import shutil
import pathlib
import csv
"""
takes 3 arguments
sys.argv[1] is the file path for the FTPdetectinfo file
sys.argv[2] is the file path for the directory containing the the FF_.fits files
sys.argv[3] is a Y/N question on whether to keep the png files generated as an intermediate and necessary step in this process

pred_list is a list of tuples of (prediction, file name)
prediction is a value between 0 and 1 that is rounded to two decimal places,
if you get a value outside of this range something is going very wrong
1 is a positive prediction that the detection is a meteor
and 0 is a negative prediction that the detection is not a meteor

dependencies:
RMS
numpy
tensorflow = 4.2.1     Unsure if >4.2.1 will break anything
and all dependencies of these packages, should be able to install numpy and tensorflow using pip or any package manager



Final activation layer of model uses a sigmoid function and so the predictions should follow that distribution,
cutoff value for manual inspection needs to be decided upon

png file names are generated so that it has the same name as the .fits file it was generated from and has a suffix "_n.png"
 where n is the detection number given to that detection as in the FTPdetectinfo file

These scripts will likely not handle errors well and will probably have problems if the files or folders are not named correctly and
I am unsure if it will handle the .bin files
"""

if len(sys.argv) == 4:
    FTP_path_1 = sys.argv[1]
    FF_dir_path_1 = sys.argv[2]
    keep_pngs = sys.argv[3]
else:
    #default options here should be changed
    FTP_path_1 = input("enter path of FTPdetectinfo file to be used: ")

    FF_dir_path_1 = input("enter path of FF_file dir to be used: ")
    keep_pngs = input("Keep temp pngs (Y/N): ")

#hardcoded path for the current last model Fiachra trained, can easily be changed with a different .h5 file so long as it has been trained using all the same preprocessing steps otherwise will required modifications to the make_pred.py script and possibly the png_gen.py script
model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "meteorml_20220325_1.h5")

png_gen.gen_pngs(FTP_path_1, FF_dir_path_1)
#little quirk, tf.keras.image_dataset_from_directory() needs argument to be directory path, this directory then needs to contain another directory which contains all the images, "labels" of images also affects this quirk
pred_list = make_pred.cust_predict(os.path.join(FF_dir_path_1, "temp_png_dir"), model_path)

#print(pred_list)

if keep_pngs not in ["Y", "y"]:
    shutil.rmtree(os.path.join(FF_dir_path_1, "temp_png_dir"))

# to rename all the png file to have the prediction at the end of the file name, delete all the # in the following segment
# try:
#     new_list = []
#     for pred_2, file_name_2 in pred_list:
#         new_file_name = file_name_2[:-4] + f"_{(pred_2 * 100):.0f}" + ".png"
#         new_list.append((pred_2, new_file_name))
#         os.rename(file_name_2, new_file_name)
#     pred_list = new_list
# except Exception as e:
#     print(f"Error occured: {e}")
#     pass



#to write the predictions to a csv file in the .fits folder, delete all the # in the following segment
# os.chdir(FF_dir_path_1)
# for detect in pred_list:
#    with open("meteorml_preds.csv", "w") as csv_logfile:
#        csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")
#        csv_logfile_writer.writerow(["Prediction", "File Name"])
#        for pred_1, file_name_1 in pred_list:
#            csv_logfile_writer.writerow([pred_1, file_name_1])

print("finished")
