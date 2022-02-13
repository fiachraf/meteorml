import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


pred_csv = input("csv file containing predictions: ")
#print("test0")
with open(pred_csv) as csv_file:
    #print("test1")
    csv_reader = csv.reader(csv_file)
    row_index = 0
    for row in csv_reader:
        #print("test2")
        #print(f"row[2]: {row[2]}")
        #row[1] is prediction, row[0] is label, row[2] is false prediction, row[3] is the file name
        #columns(row indices) might be different depending on file
        #for the first one false means a false prediction and true means a true prediction
        if row[2] == "False":
            #print("test3")
            plt.figure(figsize = (6,6))
            #ax1 = fig.add_subplot()
            #ax2 = fig.add_subplot()
            img = mpimg.imread(row[3])
            plt.imshow(img)
            #plt.subplots_adjust(top=1)
            #plt.gcf().text(0.02, 0.75, f"Label: {row[0]}, Prediction: {row[1]}", fontsize=10)
            #plt.gcf().text(0.02, 1, f"file name: {row[3]}", fontsize=10)
            plt.title(f"file name: {row[3][52:]}")
            plt.xlabel(f"Label: {row[0]}, Prediction: {row[1]}")
            #plt.rcParams["figure.figsize"] = (10, 10)
            plt.show()
