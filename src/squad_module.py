import pandas as pd
from os.path import exists
from base_dataset_module import PeanutDataSet 

class SQuAD(PeanutDataSet):
    def __init__(self):
        super().__init__()

    # read data from a json file and load it in self.data
    def load_data(self, path_to_read_from, path_to_write_to):
        squad_raw_data = pd.read_json(path_to_read_from)

        if(not exists(path_to_write_to)):
            comp_list = []
            for index, row in squad_raw_data.iterrows():
                for i in range(len(row["data"]["paragraphs"])): # each article
                    for j in range(len(row["data"]["paragraphs"][i])): # each paragraph
                        for k in range(len(row["data"]["paragraphs"][i]['qas'])):# each question about the selected paragraph
                            temp_list = []
                            temp_list.append(row["data"]["paragraphs"][i]["context"])
                            temp_list.append(row["data"]["paragraphs"][i]['qas'][k]["question"])
                            try:
                                temp_list.append(row["data"]["paragraphs"][i]['qas'][k]["answers"][0]["text"])
                            except IndexError:
                                temp_list.append("")
                            comp_list.append(temp_list)
            new_df = pd.DataFrame(comp_list, columns=self.cols) 
            new_df.to_csv(path_to_write_to, index=False)

        self.data = pd.read_csv(path_to_write_to)
        self.no_questions = len(self.data)