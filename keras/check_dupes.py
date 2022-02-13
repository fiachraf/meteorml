import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

pred_csv = input("csv file containing predictions: ")
lines_as_dicts_in_list = []

with open(pred_csv, mode="r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    row_index = 0
    for row in csv_reader:
        lines_as_dicts_in_list.append(row["File Name"])
        #row[0] is prediction, row[1] is label, row[2] is false prediction, row[3] is the file name
        #row[2] might be different depending on file
        #for the first one false means a false prediction and true means a true prediction
        # if row["True Prediction"] == "False":
        #     print("test3")
        #     print(row["File Name"])
            # img = mpimg.imread(row[3])
            # plt.imshow(img)
            # plt.xlabel(f"Prediction: {row[0]}, label: {row[1]}, file name: {row[3]}")
            # plt.show()


dupes_list = []
print("")
#dupes_list will be a list of tuples, dupes_list[0] is the filename, duspes_list[1] is the number of time it appears
len_lines_as_dicts_in_list = len(lines_as_dicts_in_list)
for index_1, line_1 in enumerate(lines_as_dicts_in_list):
    # lines_as_dicts_in_list_copy = lines_as_dicts_in_list[index_1:]
    sys.stdout.write("\r")
    sys.stdout.write(f"progress: {index_1} of {len(lines_as_dicts_in_list)}")
    sys.stdout.flush()
    index_3 = index_1 + 1
    if index_3 < len_lines_as_dicts_in_list -1:
        for index_2, line_2 in enumerate(lines_as_dicts_in_list[index_3:]):
            if os.path.basename(line_1) == os.path.basename(line_2):
                print(f"\n{line_1}\n{line_2}")
                # if os.path.basename(line_1) not in
                if not any(d["File Name"] == f"{os.path.basename(line_1)}" for d in dupes_list):
                    dupes_list.append({"File Name":os.path.basename(line_1), "count":2})
                    # print(dupes_list)
                else:
                    for entry in dupes_list:
                        if os.path.basename(line_1) == entry["File Name"]:
                            entry["count"] += 1
    # if index_1 == 1000:
    #     break
# meteorml_20220209_1_false_iden_log_singly_tf.csv
log_file_name = f"{pred_csv[:19]}_dupes.csv"
with open(log_file_name, "w") as csv_logfile:
    csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")
    csv_logfile_writer.writerow(["File Name", "count"])
    for entry_1 in dupes_list:
        csv_logfile_writer.writerow([entry_1["File Name"], entry_1["count"]])
        print(entry_1["File Name"], entry_1["count"])
# for line_3 in dupes_list:
#     print(line_3)
