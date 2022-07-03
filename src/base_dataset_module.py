import pandas as pd

class PeanutDataSet():
    def __init__(self):
        self.cols = ["text","question","answer"]


    def load_data(self, path_to_read_from, path_to_write_to):
        self.data = pd.read_json(path_to_read_from)