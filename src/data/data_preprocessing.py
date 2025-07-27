import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger=logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("preprocessing_error.log")
file_handler.setLevel(logging.ERROR)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_comment(comment):
    try:
        comment=comment.lower()
        comment=comment.strip()
        comment=re.sub('\n',' ',comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words=set(stopwords.words("english"))-{"not","but","however","yet","no"}
        comment=' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer=WordNetLemmatizer()
        comment=' '.join(lemmatizer.lemmatize(word) for word in comment.split())

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing the comment {e}")
        raise

def normalize_text(df):
    try:
        df['clean_comment']=df['clean_comment'].apply(preprocess_comment)
        logger.debug("Text Normalization completed")
        return df
    except Exception as e:
        logger.error(f"Error during text Normalization {e}")
        raise 

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:

        interim_data_path=os.path.join(data_path,"interim")
        logger.debug(f"Creating the Directory in {interim_data_path}")
        os.makedirs(interim_data_path,exist_ok=True)
        logger.debug(f"Created the directory in {interim_data_path}")

        train_data.to_csv(os.path.join(interim_data_path,"train_preprocessed.csv"),index=False)
        test_data.to_csv(os.path.join(interim_data_path,"test_preprocessed.csv",),index=False)
        logger.debug(f"Processed data saved to the path {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occured while saving the data {e}")
        raise

def main():
    try:
        logger.debug("Starting the  data preprocessing ")

        train_data=pd.read_csv("./data/raw/train.csv")
        test_data=pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded Sucessfully")

        train_processed_data=normalize_text(train_data)
        test_processed_data=normalize_text(test_data)

        save_data(train_processed_data,test_processed_data,"./data")
    except Exception as e:
        logger.error(f"Fialed to complete the data preprocessing step {e}")
        raise

if __name__=='__main__':
    main()


