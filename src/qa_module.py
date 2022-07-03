import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import pandas as pd


class PeanutQASystem():
    def __init__(self): 
        self.tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        self.model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

    def predict_answer_for_qi(self, qid, data):
        return self.predict_answer(data["question"][qid], data["text"][qid])

    def predict_answer(self, question, context):
        #encoding question + context to input a single vector to bert
        input_ids = self.tokenizer.encode(question, context)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids) # getting tokens back

        sep_idx = input_ids.index(self.tokenizer.sep_token_id)

        #number of tokens in segment Q (question)
        num_seg_q = sep_idx + 1
        #number of tokens in segment T (text)
        num_seg_t = len(input_ids) - num_seg_q

        #creating the segment ids
        segment_ids = [0]*num_seg_q + [1]*num_seg_t #to differentiate our segments - question and text

        #making sure that every input token has a segment id
        assert len(segment_ids) == len(input_ids)

        output = self.model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))

        #tokens with highest start and end scores
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        if answer_end >= answer_start:
            answer = tokens[answer_start]
            for i in range(answer_start+1, answer_end+1): #removing Bert spetials symbols like ##
                if tokens[i][0:2] == "##":
                    answer += tokens[i][2:]
                else:
                    answer += " " + tokens[i]
        else:
            answer = ""

        if answer.startswith("[CLS]"):
            answer = ""

        return answer
            
    def predict_n_answers(self, data, path_to_write_to, n =-1):
        if n == -1:
            n = len(data)
        predictions = []
        for qid in range(n):
            print("predicting answer",qid)
            question = data["question"][qid]
            context = data["text"][qid]
            prediction = self.predict_answer(question, context)
            predictions.append(prediction)
        
        # create csv with answers
        predictions_df = pd.DataFrame(predictions, columns=['prediction'])
        predictions_df.to_csv(path_to_write_to, index=False)
        self.data = pd.read_csv(path_to_write_to, keep_default_na=False)
        print("finished with predictions")

    