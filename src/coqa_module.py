import pandas as pd
from os.path import exists
from base_dataset_module import PeanutDataSet 

class CoQA(PeanutDataSet):
    def __init__(self):
        super().__init__()

    # read data from a json file and load it in self.data
    def load_data(self, path_to_read_from, path_to_write_to):
        coqa_raw_data = pd.read_json(path_to_read_from)

        if(not exists(path_to_write_to)):
            print("Create dataframe rows")
            comp_list = []
            for index, row in coqa_raw_data.iterrows():
                for i in range(len(row["data"]["questions"])):
                    temp_list = []
                    temp_list.append(row["data"]["story"])
                    temp_list.append(row["data"]["questions"][i]["input_text"])
                    temp_list.append(row["data"]["answers"][i]["input_text"])
                    comp_list.append(temp_list)
            new_df = pd.DataFrame(comp_list, columns=self.cols) 
            #saving the dataframe to csv file for further loading
            new_df.to_csv(path_to_write_to, index=False)
        self.data = pd.read_csv(path_to_write_to)
        self.no_questions = len(self.data)