# Fakenews Detection
Fake news has come into the picture since the US president election 2016. Since, then the 
problem has emerged continuously with the increased use of social media.  To contribute towards this exciting area of research, this work creates a fake news detector 
using machine learning technology. The project presents the idea of finding whether the news 
article is fake or true.
The model is trained by first performing the initial  steps of gathering, cleaning  and modifying of data. 
The first step in the implementation of the project is finding out the relevant data in the area 
of research. To accomplish the task of fact checking 
of news articles to see if its fake or not fake the requirement was to gather some labelled 
headlines. Csv files from two data sources were collected. First file factcheck.csv was taken from GitHub repository of data science project. The second file 
fake_or_real_news.csv was taken from Kaggle. 
In order to make the machine understand the data for analysis it needs to be converted into 
a machine-readable format. To accomplish this task the Bag of Words model is implemented with machine learning algorithms.
The machine learning classification algorithm giving the highest accuracy is then implemented. 
In this project, four classification algorithms logistic regression, K-nearest neighbor, Decision tree and random forest were tested against their accuracy. 
The logistic regression algorithm gave the highest accuracy of 71% and hence the model is saved with it.  
