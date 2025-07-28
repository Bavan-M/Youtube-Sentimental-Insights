import os
import numpy as np
import pandas as pd
import logging
import pickle
import yaml
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature

logger=logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("model_Evaluation_error.log")
file_handler.setLevel(logging.ERROR)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug(f"Data loaded from the path {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error occured while loading the data from the path {e}")
        raise

def load_model(file_path:str):
    try:
        with open(file_path,"rb") as file:
            model=pickle.load(file)
        logger.debug(f"Model loaded Sucessfullly from the path {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error occured while loading the model {e}")
        raise
        

def load_vectorizer(file_path:str)->TfidfVectorizer:
    try:
        with open(file_path,"rb") as file:
            vectorizer=pickle.load(file)
        logger.debug(f"Vectorizer loaded from the path {file_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Error occured while loading the vectorizer from the path{e}")
        raise

def load_params(file_path:str)->dict:
    try:
        with open(file_path,"r") as file:
            params=yaml.safe_load(file)
        logger.debug(f"Params loaded from the path {file_path}")
        return params
    except Exception as e:
        logger.error(f"Error occured while loading the Params file {e}")
        raise

def evaluate_model(model,X_test:np.ndarray,y_test:np.ndarray):
    try:
        y_pred=model.predict(X_test)
        report=classification_report(y_pred,y_test,output_dict=True)
        cn_matrix=confusion_matrix(y_pred,y_test)
        logger.debug(f"Evalating the model completed")
        return report,cn_matrix
    except Exception as e:
        logger.error(f"Error occured while Evaluating the model {e}")
        raise

def log_confusion_matrix(cm,dataset_name):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap="Blues")
    plt.title(f"Confusion matrix for {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_file_path=f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()
    
    
def save_model_info(run_id:str,model_path:str,file_path:str)->None:
    try:
        model_info={
            "run_id":run_id,
            "model_path":model_path
        }
        with open(file_path ,"w") as file:
            json.dump(model_info,file,indent=4)
        logger.debug(f"Model info saved into {file_path}")
    except Exception as e:
        logger.error(f"Error occured while saving the model info into the path {e}")
        raise

def main():
    mlflow.set_tracking_uri("http://ec2-XX-X-XXX-XXX.ap-south-1.compute.amazonaws.com:5000/")
    mlflow.set_experiment("dvc-pipeline-run")
    with mlflow.start_run() as run:
        try:
            root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../"))
            params=load_params(file_path=os.path.join(root_dir,"params.yaml"))

            for key,val in params.items():
                mlflow.log_param(key=key,value=val)

            logger.debug("Started loading the model,vectorizer and test_data")
            model=load_model(file_path=os.path.join(root_dir,"lgbm_model.pkl"))
            vectorizer=load_vectorizer(file_path=os.path.join(root_dir,"tfidf_vectorizer.pkl"))
            test_data=load_data(file_path=os.path.join(root_dir,"data/interim/test_preprocessed.csv"))
            logger.debug("Finished loading the model,vectorizer and test_data")

            logger.debug("Started vectorization transform on test_data")
            X_test_vectorized=vectorizer.transform(test_data['clean_comment'].values)
            y_test=test_data['category'].values
            logger.debug(f"Finished vectorization the shaped are {X_test_vectorized.shape} and {y_test.shape}")

            sample_tfidf_input_df=pd.DataFrame(X_test_vectorized.toarray()[:5],columns=vectorizer.get_feature_names_out())
            model_input_output_signature =infer_signature(sample_tfidf_input_df,model.predict(X_test_vectorized[:5]))

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=model_input_output_signature,
                input_example=sample_tfidf_input_df
            )

            model_path="lgm_model"
            artifact_uri = mlflow.get_artifact_uri("lgbm_model")
            save_model_info(run.info.run_id, artifact_uri, "experiment_info.json")

            mlflow.log_artifact(os.path.join(root_dir,"tfidf_vectorizer.pkl"))
            
            report,cm=evaluate_model(model=model,X_test=X_test_vectorized,y_test=y_test)
            for label,metrics in report.items():
                if isinstance(metrics,dict):
                    mlflow.log_metrics(
                        {
                            f"test_{label}_precision":metrics["precision"],
                            f"test_{label}_recall":metrics["recall"],
                            f"test_{label}f1-score":metrics["f1-score"],
                        }
                    )
            log_confusion_matrix(cm,"Test Data")

            mlflow.set_tags({
                "model_type":"LightGBM",
                "task":"Sentimental_Analysis",
                "dataset":"Youtube_Comments"
            })

        except Exception as e:
            logger.error(f"Failed to complete the model Evaluation {e}")
            print("Error {e}")
    
if __name__=="__main__":
    main()
        