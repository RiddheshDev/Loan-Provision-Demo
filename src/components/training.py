import os 
import sys
import json 
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingConfig
from src.components.exception import CustomException
from src.Pipelines.custom_pipeline import NewAppModel, OptInModel ## Import your main pipelines here
from src.Pipelines.utils import get_subpipeline
from src.entity.constants import MODEL_REGISTRY_CLF, MODEL_REGISTRY_REG , CONFIGFOLDER,PARAMFILE, MODELFOLDER, MODELREPORT, SHEETNAME, MODELCONFIG, FEATURESCORETRAIN, FEATURESCORELIVE
from src.components.data_validation import DataValidation
from src.components.hyperparameter_tuning import HyperparameterTuning
from src.Pipelines.utils import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix
from typing import Optional
import cloudpickle
import traceback
import warnings
warnings.filterwarnings('ignore')
import argparse

class ModelTraining:
    def __init__(self,training_config:TrainingConfig,preprocessor:Pipeline,models_registry:dict,models_config:Optional[dict] = None,df=None,):
        self.training_config = training_config
        self.preprocessor = preprocessor
        self.dataframe = df
        self.models_config = self.__class__.get_models_config(models_config=models_config)
        self.model_registry = models_registry
        self.result_path = os.path.join(self.training_config.run_path,MODELFOLDER,MODELREPORT)
        self.model_folder = os.path.join(self.training_config.run_path,MODELFOLDER)

    def preprocessing_data(self,X:pd.DataFrame,Y:pd.DataFrame):
        x_processed,y_processed = self.preprocessor.fit_resample(X,Y)
        return x_processed,y_processed

    @staticmethod
    def get_models_config(models_config:Optional[dict]):
        if models_config is None:
            with open(os.path.join(CONFIGFOLDER,MODELCONFIG),'r',encoding='utf-8') as file:
                return json.load(file)
        return models_config

    def get_model(self,):
        print("Selecting Model",flush=True)
        if len(self.models_config) > 1 :
            model_name = max(self.models_config,key=lambda k: self.models_config[k].get('best_score'))
            print(model_name)
            model_params = self.models_config[model_name].get('best_params')
            print(model_params)
            return model_name,model_params
        
        elif len(self.models_config) ==1 :
            model_name = next(iter(self.models_config))
            model_params = self.models_config[model_name].get('best_params')
            return model_name, model_params
        
        else :
            raise ValueError(f"No model is mentioned in {MODELCONFIG}")

    @staticmethod
    def create_pipeline(preprocessor:Pipeline,model):
        temp_pipeline =[]
        for name,step in preprocessor.steps:
            temp_pipeline.append((name,step))
        temp_pipeline.append(("model",model))
        new_pipeline = Pipeline(temp_pipeline)
        return new_pipeline
    
    def get_extra_output(self,model,x_train,y_train,config_folder,feature_score_file,sheet_name
        ):
        """
        Trains bootstrap models, computes feature scores (if available),
        and saves outputs to disk.
        """
        # ---- Train bootstrap models ----
        boot_strap_models = train_bootstrap_models(model,preprocess=get_subpipeline(self.preprocessor, resampling_enable=False),
            features=get_features(self.preprocessor),x_train=x_train,y_train=y_train)
        
        # ---- Feature score handling ----
        feature_score_path = os.path.join(config_folder, feature_score_file)
        feature_score = None

        if os.path.exists(feature_score_path):
            with open(feature_score_path, "r") as f:
                data = json.load(f)

            if sheet_name.lower() not in data:
                raise KeyError(f"'{sheet_name.lower()}' not found in feature score file")

            feature_score = compute_feature_distributions(
                x_train,
                data[sheet_name.lower()]
            )

            with open(os.path.join(self.model_folder, FEATURESCORELIVE), "w") as f:
                json.dump(feature_score, f, indent=2)

        # ---- Save bootstrap models ----
        model_path = os.path.join(self.model_folder, f"bootstrap_{sheet_name}.pkl")
        with open(model_path, "wb") as file:
            cloudpickle.dump(boot_strap_models, file)

        return boot_strap_models, feature_score

    def train_model(self,X:pd.DataFrame,Y:pd.DataFrame,extra_output:bool=False):
        model_name , model_params = self.get_model()
        print(f"Model Selected {model_name}")
        model = self.model_registry.get(model_name)
        if  model is None:
            raise ValueError(f"Model not found : {model_name}")
        
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
        # x_train,y_train = self.preprocessing_data(X=x_train,Y=y_train)
        model.set_params(**model_params)
        final_pipeline = self.__class__.create_pipeline(self.preprocessor,model)
        print("Fitting Model")
        final_pipeline.fit(x_train,y_train)
        print("Model Fitted")

        y_pred = final_pipeline.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        recall =  recall_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)
        results = {
            "accuracy":accuracy,
            "recall" : recall,
            "precision":precision,
        }
        
        os.makedirs(self.model_folder)
        

        ##saving result of model
        with open(self.result_path,'w') as file:
            json.dump(results,file,indent=4)
        model_path = os.path.join(self.model_folder,f'{model_name}_{SHEETNAME}.pkl')
        
        ##Saving The main model
        with open(model_path,'wb') as file:
            cloudpickle.dump(final_pipeline,file)

        ## Extra output paths
        if extra_output:
            boot_strap_models,feature_score = self.get_extra_output(model=model,x_train=x_train,y_train=y_train,
                                                                    config_folder=CONFIGFOLDER,feature_score_file =FEATURESCORETRAIN,sheet_name=SHEETNAME)
            # boot_strap_models = train_bootstrap_models(model,preprocess=get_subpipeline(self.preprocessor,resampling_enable=False),
            #                                            features=get_features(self.preprocessor),x_train=x_train,y_train=y_train)
            # feature_score_path = os.path.join(CONFIGFOLDER,FEATURESCORETRAIN)

            # if os.path.exists(feature_score_path):
            #     with open(feature_score_path, "r") as f:
            #         data = json.load(f)
            #     feature_score = compute_feature_distributions(x_train,data[SHEETNAME.lower()])
            #     with open(os.path.join(self.model_folder,FEATURESCORELIVE), "w") as f:
            #         json.dump(feature_score, f, indent=2)
            # else :
            #     feature_score = None
            # ##save Bootstrap Models
            # with open(os.path.join(self.model_folder,f'bootstrap_{SHEETNAME}.pkl'),'wb') as file:
            #     cloudpickle.dump(boot_strap_models,file)   

        return results

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Paramters for training")
    parser.add_argument('--tuning',action="store_true",help="Enable Hyperparameter Tuning")
    args = parser.parse_args()
    training_config = TrainingConfig()
    df = DataIngestion(training_config=training_config).read_data()
    data_validation = DataValidation(training_config=training_config).initiate_data_validation(df=df)
    if not data_validation:
        print("Code stopping",flush=True)
        sys.exit(1)
    print('Code continuing')
    X,Y = data_validation[1],data_validation[2]
    ## Defininf models and pipeline
    main_pipeline = NewAppModel().get_model_pipeline()
    preprocessor_pipeline = get_subpipeline(pipe=main_pipeline)
    best_models=None
    if args.tuning:
        print('Parameter tuning')
        best_models = HyperparameterTuning(training_config=training_config,models=MODEL_REGISTRY_CLF,preprocessor=preprocessor_pipeline).run_multiple_model_tuning(X=X,Y=Y)
        # print(best_models)
    print('Training Model')
    
    results = ModelTraining(training_config,preprocessor_pipeline,MODEL_REGISTRY_CLF,best_models).train_model(X,Y,extra_output=True)
    print(results)
