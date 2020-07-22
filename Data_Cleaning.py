import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_string_dtype

general_news = 'C://Users//Vanya//PycharmProjects//Fakenews//Replacedfile.csv'
generalnews_training_file = 'generalnews_training_set.csv'
generalnews_test_file = 'generalnews_test_set.csv'

coronavirus_news = 'C://Users//Vanya//PycharmProjects//Fakenews//Coronanews.csv'
coronavirus_training_file = 'coronavirus_training_set.csv'
coronavirus_test_file =  'coronavirus_test_set.csv'



def remove_non_ascii(text):
   i =int()
   return ''.join(i for i in text if ord(i)<128)


def clean_data(data_frame):
    # Clean the data
    data_frame.dropna()
    data_frame.drop_duplicates(keep='first', inplace=False) # Removing duplicates

    for column in data_frame:
        if(is_string_dtype(data_frame[column])):
            data_frame[column] = data_frame[column].apply(remove_non_ascii)

    return data_frame


if __name__ == "__main__":
    # Load the Data
    news_df = pd.read_csv(coronavirus_news, dtype={'News':str,'Label':bool},na_values=" ")
    news_df = clean_data(news_df)

    # Split the training & testing dataset in 80:20 ratio
    train, test = train_test_split(news_df, test_size=0.2)
    train.to_csv(coronavirus_training_file)
    test.to_csv(coronavirus_test_file)

