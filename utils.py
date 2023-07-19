import csv
from datetime import datetime


def save_log(log, log_dir):
    # store log object as a CSV
    filename = "log_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = log_dir + f"/{filename}.csv"
    log_file = open(file_path, "w")
    log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    columns = list(log[0].keys())
    log_writer.writerow(columns)
    for line in log:
        values = [line[c] for c in columns]
        log_writer.writerow(values)
    log_file.close()
