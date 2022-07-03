import string, re
import pandas as pd
from datetime import datetime
import os

class Validator():
    def __init__(self):
        pass

    def get_prediction(self, qid, data):
        prediction = data["prediction"][qid]

        if not prediction:
            prediction = ""
            
        return prediction
    
    # Retrieves all possible true answers the dataset proposes
    def get_dataset_answers(self, qid, data):
        gold_answers = [data["answer"][qid]]

        # empty string 
        if not gold_answers:
            gold_answers = [""]
            
        return gold_answers

    def remove_articles(self, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(self, text):
        return " ".join(text.split())

    def remove_punc(self, text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(self, text):
        return text.lower()

    # Removing articles and punctuation and standardizing whitespace
    def normalize_text(self, s):
        return self.white_space_fix(self.remove_articles(self.remove_punc(self.lower(s))))

    def calculate_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))

    def calculate_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()
        
        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0
        
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        
        return 2 * (prec * rec) / (prec + rec)

    def calculate_metrics(self, question_data, predictions_data, n, save_results=False, path_to_write_to=""):
        exact_metrics_list = {}
        f1_metrics_list = {}
        question_ids_list = []
        for qid in range(n):
            prediction = self.get_prediction(qid, predictions_data)
            gold_answers = self.get_dataset_answers(qid, question_data)
            em_score = max((self.calculate_exact_match(prediction, answer)) for answer in gold_answers)
            f1_score = max((self.calculate_f1(prediction, answer)) for answer in gold_answers)
            exact_metrics_list[qid] = em_score
            f1_metrics_list[qid] = f1_score
            question_ids_list.append(qid)

        if(save_results): #create file with results
            evaluations_df = pd.DataFrame(list(zip(exact_metrics_list.values(), f1_metrics_list.values())),
               columns =['exactMatch', 'f1'])
            evaluations_df.to_csv(path_to_write_to, index=False)
            self.data = pd.read_csv(path_to_write_to, keep_default_na=False)

        return self.calculate_exact_match_and_f1score(exact_metrics_list, f1_metrics_list, question_ids_list, save_results, path_to_write_to)

    def calculate_exact_match_and_f1score(self, exact_metrics_list, f1_metrics_list, question_ids_list, save_file = False, path_to_write_to=""):
        total = -1
        exact = -1
        f1 = -1
        if not question_ids_list:
            total = len(exact_metrics_list)
            exact = 100.0 * sum(exact_metrics_list.values()) / total
            f1 = 100.0 * sum(f1_metrics_list.values()) / total
        else:
            total = len(question_ids_list)
            exact= 100.0 * sum(exact_metrics_list[k] for k in question_ids_list) / total
            f1 = 100.0 * sum(f1_metrics_list[k] for k in question_ids_list) / total
            total= total
        if save_file :
            correct_path_to_write_to = path_to_write_to.replace("./", "")
            correct_path_to_write_to = path_to_write_to.replace(".csv", "")
            correct_path_to_write_to = correct_path_to_write_to.replace("/", "\\")
            cwd = os.getcwd()
            complete_path = os.path.join(cwd, correct_path_to_write_to + ".txt")
            _file = open(complete_path, "w+")

            # datetime object containing current date and time
            now = datetime.now()

            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

            _file.write("date: " + str(dt_string) + "\r\n\n")
            _file.write('total: ' + str(total) + "\r\n\n")
            _file.write('exact match: '+ str(exact) + "\r\n\n")
            _file.write('f1 score: '+ str(f1) + "\r\n\n")

            _file.close()
        print("finished metrics evaluation step")
        return (exact, f1, total)