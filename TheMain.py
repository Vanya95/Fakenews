import pickle
import pandas as pd

load_model = pickle.load(open('model.sav', 'rb'))  # Loading model


def classifyNews(var):
    predictedOp = load_model.predict([var])  # Predicting
    prob = load_model.predict_proba([var])  # Probability estimation
    return (print("Stmt ",predictedOp[0]),
        print("Truth likelihood ",prob[0][1]))
if __name__ == '__main__':

        iptext = input("Enter news headline:  ")
        print("Entered news is: " + str(iptext))
        classifyNews(iptext)

