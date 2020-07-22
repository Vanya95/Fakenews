import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

coronavirus_training_file = 'coronavirus_training_set.csv'
coronavirus_test_file = 'coronavirus_test_set.csv'
generalnews_training_file = 'generalnews_training_set.csv'
generalnews_test_file = 'generalnews_test_set.csv'

def confusion_matrix_plot(
        c,classes,normalize=False,title = 'Confusion Matrix',cmap = plt.cm.Blues):
        plt.imshow(c,interpolation='nearest',cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes)
        plt.yticks(tick_marks,classes)
        for i in range(c.shape[0]):

                for j in range(c.shape[1]):(
                        plt.text(j,i,format(c[i,j]))
                )


        plt.tight_layout()
        plt.xlabel('Actual')
        plt.ylabel('predicted')

        plt.show()

train= pd.read_csv(coronavirus_training_file,dtype=object)
test = pd.read_csv(coronavirus_test_file,dtype=object)
actual_data = test

countV = CountVectorizer()
logRegressionP = Pipeline([
        ('LogRCV', countV),
        ('LogR_clf',LogisticRegression())
        ])
logRegressionP.fit(train['News'], train['Label'])
predicted_LogR = logRegressionP.predict(test['News'])
print('Accuracy for Count Vectorizer with logistic regression:', np.mean(predicted_LogR == test['Label']))

logRP = Pipeline([
        ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])
logRP.fit(train['News'], train['Label'])
predicted_LogR_ngram = logRP.predict(test['News'])
print('Accuracy for Tfidf Vectorizer with logistic regression', np.mean(predicted_LogR_ngram == test['Label']))
c = confusion_matrix(test['Label'],predicted_LogR_ngram,labels=['False','True'])
confusion_matrix_plot(c,classes=['False','True'])


logH = Pipeline([
        ('LogHv',HashingVectorizer(decode_error='ignore',stop_words='english',ngram_range=(1,10),n_features=1048576, binary=False)),
        ('LogR_clf',LogisticRegression())
                ])
logH.fit(train['News'],train['Label'])
predicted_Hv = logH.predict(test['News'])
print('Accuracy for Hashing Vectorizer with logistic Regression', np.mean(predicted_Hv == test['Label']))

KnnP = Pipeline([
        ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
        ('LogR_clf',KNeighborsClassifier(n_neighbors=5))
        ])

KnnP.fit(train['News'], train['Label'])
predict_knn = KnnP.predict(test['News'])
print('Accuracy for Tfidf Vectorizer with KNN', np.mean(predict_knn == test['Label']))
# predict = predict_knn.astype(int).flatten()
#predictedvalue = np.array(predict)
predicted_value = np.array(predict_knn == test['Label'])

KnnPr = Pipeline([
        ('LogRCV', countV),
        ('LogR_clf',KNeighborsClassifier())
        ])
KnnPr.fit(train['News'], train['Label'])
predicted_knn = KnnPr.predict(test['News'])
print('Accuracy for Count Vectorizer with KNN:', np.mean(predicted_knn == test['Label']))

KnnH = Pipeline([
        ('LogHv',HashingVectorizer(decode_error='ignore',stop_words='english',ngram_range=(1,10),n_features=1048576, binary=False)),
        ('LogR_clf',KNeighborsClassifier())
                ])
KnnH.fit(train['News'],train['Label'])
predi_knn = KnnH.predict(test['News'])
print('Accuracy for Hashing Vectorizer with KNN', np.mean(predi_knn == test['Label']))

DeciP = Pipeline([
        ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
        ('LogR_clf',DecisionTreeClassifier())
        ])

DeciP.fit(train['News'], train['Label'])
predicted_deci = DeciP.predict(test['News'])
print('Accuracy for Tfidf Vectorizer with Decision tree:', np.mean(predicted_deci == test['Label']))

DecisionP = Pipeline([
        ('LogRCV', countV),
        ('LogR_clf',DecisionTreeClassifier())
        ])
DecisionP.fit(train['News'], train['Label'])
predicted_decision = DecisionP.predict(test['News'])
print('Accuracy for Count Vectorizer with Decision tree:', np.mean(predicted_decision == test['Label']))

DeciH = Pipeline([
        ('LogHv',HashingVectorizer(decode_error='ignore',stop_words='english',ngram_range=(1,10),n_features=1048576, binary=False)),
        ('LogR_clf',DecisionTreeClassifier())
                ])
DeciH.fit(train['News'],train['Label'])
predi_Deci = DeciH.predict(test['News'])
print('Accuracy for Hashing Vectorizer with Decision tree', np.mean(predi_Deci == test['Label']))

RandomP = Pipeline([
        ('LogRCV', countV),
        ('LogR_clf',RandomForestClassifier())
        ])
RandomP.fit(train['News'], train['Label'])
predicted_random = RandomP.predict(test['News'])
print('Accuracy for Count Vectorizer with Random forest', np.mean(predicted_random == test['Label']))

RFP = Pipeline([
        ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
        ('LogR_clf',RandomForestClassifier())
        ])

RFP.fit(train['News'], train['Label'])
predicted_RFP = RFP.predict(test['News'])
print('Accuracy for Tfidf Vectorizer with Random forest', np.mean(predicted_RFP == test['Label']))


RanFP = Pipeline([
        ('LogHv',HashingVectorizer(decode_error='ignore',stop_words='english',ngram_range=(1,10),n_features=1048576, binary=False)),
        ('LogR_clf',RandomForestClassifier())
                ])
RanFP.fit(train['News'],train['Label'])
predicted_RP = RanFP.predict(test['News'])
print('Accuracy for Hashing Vectorizer with Random forest', np.mean(predicted_RP == test['Label']))

# Saving the model
model_file = 'model.sav'
pickle.dump(logRP, open(model_file, 'wb'))
print("Dumped the file")

# Actual vs Predicted


print("\t","\t",actual_data.head(10), "\t","\t",predicted_value)