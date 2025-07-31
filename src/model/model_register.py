import json
import os
import mlflow
import logging

mlflow.set_tracking_uri("http://xxx-xx-xxx-xxx-xxx.xx-xxxx-x.compute.amazonaws.com:5000/")


logger=logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("model_Registration_error.log")
file_handler.setLevel(logging.ERROR)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(filpath:str)->dict:
    try:
        with open(filpath,"r") as file:
            model_info=json.load(file)
        logger.debug(f"Sucessfully loaded the model_info from the path {filpath}")
        return model_info
    except FileNotFoundError as e:
        logger.error(f"File not found {e}")
        raise
    except Exception as e:
        logger.error(f"Error ocured while loading the model info {e}")
        raise

def register_model(model_name:str,model_info:dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version=mlflow.register_model(model_uri,model_name)

        client=mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name= model_name,version=model_version.version,stage="Staging")
        
        logger.debug(f"model {model_name} with version {model_version.version} is registered and transitioned to Staging")
    except Exception as e:
        logger.error(f"Error during model registration {e}")
        raise

def main():
    try:
        model_info_path="experiment_info.json"
        model_info=load_model_info(model_info_path)
        model_name="youtube_chromes_plugin_model1"
        register_model(model_name,model_info)
    except Exception as e:
        logger.error(f"Failed to complete the model registration {e}")
        raise

if __name__=="__main__":
    main()

