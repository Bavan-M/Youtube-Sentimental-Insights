import numpy as np
import pandas as pd
import os
import pickle
import yaml 
import logging
import lightgbm as lgbm
from sklearn.feature_extraction.text import TfidfVectorizer

logger=logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("model_Building_error.log")
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
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: {e}')
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def apply_TfidfVectorizer(train_data:pd.DataFrame,maxfeature:int,ngramrange:tuple)->tuple:
    try:
        vectorizer=TfidfVectorizer(max_features=maxfeature,ngram_range=ngramrange)
        X_train=train_data['clean_comment'].values
        y_train=train_data['category'].values

        X_train_vectorized = vectorizer.fit_transform(X_train)
        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_vectorized.shape}")

        with open(os.path.join(get_root_directory(),'tfidf_vectorizer.pkl'),"wb") as f:
            pickle.dump(vectorizer,f)

        logger.debug("TF-IDF applied and transformed")
        return X_train_vectorized,y_train
    except Exception as e:
        logger.error(f"Error occured during apply_TfidfVectorizer {e}")
        raise 

def train_lightgbm(X_train:np.ndarray,y_train:np.ndarray,learning_rate:float,max_depth:int,n_estimators:int)->lgbm.LGBMClassifier :
    try:
        model=lgbm.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train,y_train)
        logger.debug("lightgbm model training completed")
        return model
    except Exception as e:
        logger.error(f"Error occured during lightgbm model training {e}")
        raise 

def save_model(model,file_path:str)->None:
    try:
        with open(file_path,"wb") as f:
            pickle.dump(model,f)
        logger.debug(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error ocuured while saving the model {e}")
        raise

def main():
    try:
        root_dir=get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        train_data = load_data(os.path.join(root_dir, 'data/interim/train_preprocessed.csv'))

        X_train_tfidf, y_train = apply_TfidfVectorizer(train_data, max_features, ngram_range)

        best_model = train_lightgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()