import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import numpy as np
import random

pred_csv = input("csv file containing predictions: ")


with open(pred_csv, mode="r") as csv_file:
    csv_reader = csv.reader(csv_file)
    row_index = 0
    false_list = []
    for row in csv_reader:
        #print("test2")
        #print(f"row[2]: {row[2]}")
        #row[1] is prediction, row[0] is label, row[2] is false prediction, row[3] is the file name
        #columns(row indices) might be different depending on file
        #for the first one false means a false prediction and true means a true prediction
        if row[2] == "False":
            false_list.append(row)

shuffle = True

if shuffle == True:
    random.shuffle(false_list)

for label, prediction, false_status, file_name in false_list:
    #print("test3")
    plt.figure(figsize = (6,6))
    #ax1 = fig.add_subplot()
    #ax2 = fig.add_subplot()
    img = mpimg.imread(file_name)
    plt.imshow(img)
    #plt.subplots_adjust(top=1)
    #plt.gcf().text(0.02, 0.75, f"Label: {row[0]}, Prediction: {row[1]}", fontsize=10)
    #plt.gcf().text(0.02, 1, f"file name: {row[3]}", fontsize=10)
    plt.title(f"file name: {file_name[52:]}")
    plt.xlabel(f"Label: {label}, Prediction: {prediction}")
    #plt.rcParams["figure.figsize"] = (10, 10)
    plt.show()
