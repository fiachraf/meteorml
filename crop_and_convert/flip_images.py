import PIL
from PIL import image
import os

folder_input = input("folder containing images that you want to flip: ")
num_flips = int(input("enter 1 to perform just flip from left to right\nenter 2 to perform just a flip upside down\nenter 3 to flip left to right and upside down: " ))

file_list = os.listdir(folder_input)

for item in file_list:
    #read the image
    im = Image.open(item)

    if num_flips == 1:
        #mirror across vertical axis and save image
        out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        out.save(item[:-4] + "LR" + ".png")
    elif num_flips == 2:
        #mirror across horizontal axis and save image
        out = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        out.save(item[:-4] + "UD" + ".png")
    elif num_flips == 3:
        #mirror across vertical axis and save image
        inter = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        #mirror across horizontal axis and save image
        out = inter.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        out.save(item[:-4] + "LRUD" + ".png")
