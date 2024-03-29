import csv as csv
import logging

class SubmitCsv:
    # constructor
    def __init__(self, file_path):
        self.file_path = file_path

    # csv出力
    def to_csv(self,param_header,param):
        submit_file = open(self.file_path, "w", newline="")
        file_object = csv.writer(submit_file)
        file_object.writerow(param_header)
        file_object.writerows(param)
        submit_file.close()

