import csv
import datetime
import os

#logger_call_no is the nth instance of the logger function call that appears in the code. Makes it easy to search the code just need to to find logger.log call number nl
def log(log_file_name="log.csv", log_priority="", log_type="", logger_call_no=0, details="", log_time=datetime.datetime.now()):
    if log_file_name[-4:] != ".csv":
        log_file_name = log_file_name + ".csv"

    if os.path.isfile(log_file_name) == False:
        with open(log_file_name, "a") as csv_logfile:
            csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")

            csv_logfile_writer.writerow(["time", "priority", "type", "logger call no", "details"])
            csv_logfile_writer.writerow([log_time, log_priority, log_type, logger_call_no, details])

    else:
        with open(log_file_name, "a") as csv_logfile:
            csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")
            csv_logfile_writer.writerow([log_time, log_priority, log_type, logger_call_no, details])

    return


if __name__ == "__main__":
    print("created test log file")
    log("test_log")
