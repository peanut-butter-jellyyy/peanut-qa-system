from coqa_module import CoQA
from squad_module import SQuAD
from qa_module import PeanutQASystem
from evaluation import Validator
import sys


def routine_1_CoQA(tests_to_run):
    coqa_instance = CoQA() # load CoQA dataset
    coqa_instance.load_data('./CoQA/coqa-dev-v1.0.json','./CoQA/CoQA_data.csv')
    peanut_qa = PeanutQASystem()

    if(tests_to_run == -1 ):
        tests_to_run = coqa_instance.no_questions

    peanut_qa.predict_n_answers(coqa_instance.data, './CoQA/CoQA_predictions.csv', tests_to_run)

    validator = Validator()

    results = validator.calculate_metrics(coqa_instance.data, peanut_qa.data, tests_to_run, save_results= True, path_to_write_to='./CoQA/CoQA_metrics.csv')

    print("--CoQA--")
    print("Results:")
    print("(exact match, f1 score, total cases)")
    print(results)


def routine_2_SQuAD(tests_to_run):
    squad_instance = SQuAD() # load CoQA dataset
    squad_instance.load_data('./SQuAD/dev-v2.0.json','./SQuAD/SQuAD_data.csv')
    peanut_qa = PeanutQASystem()

    if(tests_to_run == -1 ):
        tests_to_run = squad_instance.no_questions

    peanut_qa.predict_n_answers(squad_instance.data, './SQuAD/SQuAD_predictions.csv', tests_to_run)

    validator = Validator()

    results = validator.calculate_metrics(squad_instance.data, peanut_qa.data, tests_to_run, save_results= True, path_to_write_to='./SQuAD/SQuAD_metrics.csv')

    print("--SQuAD--")
    print("Results:")
    print("(exact match, f1 score, total cases)")
    print(results)



if "__main__" == __name__:
    if len(sys.argv) > 3:
        print(
            "Unnexpected number or arguments. Please provide dataset to a analize and number of questions to test (1-len(dataset))"
        )
    else:
        dataset = 1
        no_test = 500
        if len(sys.argv) == 2:
            _, no_test = sys.argv
        elif len(sys.argv) == 3:
            _, no_test, dataset = sys.argv

        if (dataset == 1):
            print("Entering routine with CoQA dataset")
            routine_1_CoQA(no_test)
        else:
            print("Entering routine with SQuAD dataset")
            routine_2_SQuAD(no_test)



