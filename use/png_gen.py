import os
import sys
from PIL import Image
import traceback
import crop_funcs
from pathlib import Path


# I didn't have RMS installed in the environment that I was working in initially cos I didn't want to build opencv so this was my workaround
try:
    # from RMS.Formats import FFfile
    from RMS.Formats import FTPdetectinfo
except:
    try:
        # temporarily adds RMS to path but this is specific to my machine
        path_root = Path(__file__).parents[1]
        print(f"path_root: {path_root}/RMS")
        sys.path.insert(1, str(path_root) + "/RMS")

        # from RMS.Formats import FFfile
        from RMS.Formats import FTPdetectinfo
    except:
        print("failed to import RMS\nquitting")
        sys.exit()

def gen_pngs(FTP_path, FF_dir_path):

    os.chdir(FF_dir_path)

    #creating new directories for the png versions of ConfirmedFiles and RejectedFiles
    if "temp_png_dir" not in FF_dir_path:
        try:
            os.mkdir("temp_png_dir")
            os.mkdir("temp_png_dir/1")
        except:
            pass

    temp_png_dir = os.path.join(FF_dir_path, "temp_png_dir/1")
    print(temp_png_dir)

    try:
        #read data from FTPdetectinfo file
        FTP_file = FTPdetectinfo.readFTPdetectinfo(os.path.dirname(FTP_path), os.path.basename(FTP_path))
        fits_file_list = os.listdir(FF_dir_path)

        #loop through each image entry in the FTPdetectinfo file and analyse each image
        for detection_entry in FTP_file:
            fits_file_name = detection_entry[0]
            meteor_num = detection_entry[2]

            if fits_file_name in fits_file_list:
                square_crop_image = crop_funcs.crop_detections(detection_entry, FF_dir_path)

                #save the Numpy array as a png using PIL
                im = Image.fromarray(square_crop_image)
                im = im.convert("L")    #converts to grescale
                im.save(temp_png_dir + "/" + fits_file_name[:-5] + "_" + str(int(meteor_num)) + ".png")

            else:
                print(f"file: {fits_file_name} not found")



    except:
        print(traceback.format_exc())

    return

if __name__ == "__main__":
    if len(sys.argv) != 1:
        FTP_path_1 = sys.argv[1]
        FF_dir_path_1 = sys.argv[2]
    else:
        FTP_path_1 = input("enter path of FTPdetectinfo file to be used: ")
        FF_dir_path_1 = input("enter path of FF_file dir to be used: ")

    gen_pngs(FTP_path_1, FF_dir_path_1)
    print("Cropping Script Finished")
