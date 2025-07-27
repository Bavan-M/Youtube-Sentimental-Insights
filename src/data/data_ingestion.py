import os
import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split

logger=logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)-> dict:
    try:
        with open(params_path,"r") as file:
            params=yaml.safe_load(file)
        logger.debug(f'Pramters retrived from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'File not found {params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML error {e}')
        raise 
    except Exception as e:
        logger.error(f'Unexpected error {e}')
        raise

def load_data(data_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        logger.debug(f'Data loaded from {data_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the csv file {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occured form while loading the data from {data_path}')
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df=df[df['clean_comment'].str.strip()!='']
        logger.debug('Sucessfully pre-processed the data,Handled missing value,Removed duplicates and trimmed the whitespace')
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error Occured during preprocessing the data {e}')
        raise 
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)

        logger.debug(f"Train and Test data are saved to path {raw_data_path}")

    except Exception as e:
        logger.error(f"Unexpected error ouccured while saving the data {e}")
        raise

def main():
    try:
        params=load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../params.yaml'))
        test_size=params["data_ingestion"]["test_size"]
        df=load_data(data_path='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        preprocessed_data=preprocess_data(df)
        train_data,test_data=train_test_split(preprocessed_data,test_size=test_size,random_state=42)
        save_data(train_data=train_data,test_data=test_data,data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../data"))
    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process {e}")
        raise

if __name__=="__main__":
    main()

        