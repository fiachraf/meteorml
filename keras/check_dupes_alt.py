import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

csv_1 = input("csv file containing file names from 1 directory: ")
csv_0 = input("csv file containing file names from 0 directory; ")
lines_as_dicts_in_list = []

list_1 = []
list_0 = []

with open(csv_1, mode="r") as csv_file_1:
    csv_reader_1 = csv.reader(csv_file_1)
    row_index = 0
    for row in csv_reader_1:
        list_1.append(row[0])

with open(csv_0, mode="r") as csv_file_0:
    csv_reader_0 = csv.reader(csv_file_0)
    row_index = 0
    for row in csv_reader_0:
        list_0.append(row[0])

#print(list_1[0])
#print(list_0[0])
#print(type(list_0[0]))
#print(list_1[22906])


dupes_list = []
print("")
#dupes_list will be a list of tuples, dupes_list[0]
len_list = len(list_1)
for index_1, line_1 in enumerate(list_1):
    #print(line_1)
    # lines_as_dicts_in_list_copy = lines_as_dicts_in_list[index_1:]
    sys.stdout.write("\r")
    sys.stdout.write(f"progress: {index_1} of {len(list_1)}")
    sys.stdout.flush()
    if line_1 in list_0:
        dupes_list.append(line_1)
        print(line_1)
    #for index_0, line_0 in enumerate(list_0):
    #    if line_1 == line_0:
    #        dupes_list.append(line_1)
    #        print(line_1)

    # if index_1 == 1000:
    #     break
# meteorml_20220209_1_false_iden_log_singly_tf.csv
log_file_name = f"alt_dupes.csv"
with open(log_file_name, "w") as csv_logfile:
    csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")
    csv_logfile_writer.writerow(["File Name"])
    for entry_1 in dupes_list:
        csv_logfile_writer.writerow([entry_1])
        
