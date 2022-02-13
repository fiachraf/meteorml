import csv
import matlpotlib.pyplot as plt
import matplotlib.image as mpimg


pred_csv = input("csv file containing predictions: ")

with open(pred_csv, mode="r") as csv_file:
    csv_reader = csv.reader(csv_file)
    row_index = 0
    for row in csv_reader:
        #row[0] is prediction, row[1] is label, row[2] is false prediction, row[3] is the file name
        #row[2] might be different depending on file
        #for the first one false means a false prediction and true means a true prediction
        if row[3] == "False":
            img = mpimg.imread(row[3])
            plt.imshow(img)
            plt.xlabel(f"Prediction: {row[0]}, label: {row[1]}, file name: {row[3]}")
            plt.show()
