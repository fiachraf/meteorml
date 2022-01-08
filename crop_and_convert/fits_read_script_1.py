from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys


#to change current working directory
try:
    os.chdir("/home/fiachra/Downloads/Meteor_Files/ConfirmedFiles")
    home_dir = os.getcwd()
    os.chdir("/home/fiachra/Downloads/Meteor_Files/ConfirmedFiles/BE0001_20181209_161939_640835")

except Exception as z1:
    print("failed to change directory, exiting script without doing anything")
    sys.exit()


# with open("FTPdetectinfo_003830_20181209_161939_640835.txt", "r") as detect_file:
#     detect_file_lines_list = detect_file.readlines()
#     for line in detect_file_lines_list:
#         print(line)

with fits.open("FF_BE0001_20181209_222632_402_0544000.fits") as meteor_image:
    print(f"meteor_image.info():")
    meteor_image.info()
    print(f"meteor_image[0].header: \n {meteor_image[0].header}")
    print(f"meteor_image[1].header: \n {meteor_image[1].header}")
    print(f"meteor_image[2].header: \n {meteor_image[2].header}")
    print(f"meteor_image[3].header: \n {meteor_image[3].header}")
    print(f"meteor_image[4].header: \n {meteor_image[4].header}")


    print(f"meteor_image[0].data: \n {meteor_image[0].data}")
    # print(f"meteor_image[1].data.shape: {meteor_image[1].data.shape}")
    # print(f"meteor_image[1].data: {meteor_image[1].data}")
#    print(f"meteor_image[2].data: {meteor_image[2].data}")
#    print(f"meteor_image[3].data: {meteor_image[3].data}")
#    print(f"meteor_image[4].data: {meteor_image[4].data}")

    #code to plot the image using matplotlib
    # plt.imshow(meteor_image[0].data)
    # plt.show()

    plt.imshow(meteor_image[1].data)
    plt.show()

    plt.imshow(meteor_image[2].data)
    plt.show()

    plt.imshow(meteor_image[3].data)
    plt.show()

    plt.imshow(meteor_image[4].data)
    plt.show()


    print("\n")
