import pandas as pd
read_fact_check = pd.read_csv('C://Users//Vanya//PycharmProjects//Fakenews//ref//fact-check.csv')
read_fact_check['Label'] = read_fact_check['Label'].str.replace('half-true', 'TRUE')
read_fact_check['Label'] = read_fact_check['Label'].str.replace('mostly-true', 'TRUE')
read_fact_check['Label'] = read_fact_check['Label'].str.replace('barely-true', 'FALSE')
read_fact_check['Label'] = read_fact_check['Label'].str.replace('pants-fire', 'FALSE')

read_real_fake = pd.read_csv('C://Users//Vanya//PycharmProjects//Fakenews//ref//real_fake_news_data.csv')
read_real_fake['Label'] = read_real_fake['Label'].str.replace('FAKE','FALSE')
read_real_fake['Label'] = read_real_fake['Label'].str.replace('REAL','TRUE')

con = [read_fact_check,read_real_fake]
result = pd.concat(con)
print(result.head())

result.to_csv('Replacedfile.csv',sep=',',na_rep='',float_format='str')
