import os 
import sys
import json 
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingConfig
from src.components.exception import CustomException
from src.Pipelines.custom_pipeline import NewAppModel, OptInModel ## Import your main pipelines here
from src.Pipelines.utils import get_subpipeline, get_preprocessor_subpipeline
from src.entity.constants import MODEL_REGISTRY_CLF, MODEL_REGISTRY_REG , CONFIGFOLDER,PARAMFILE
from src.components.data_validation import DataValidation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix,recall_score
from imblearn.pipeline import Pipeline
import mlflow
import traceback
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuning:
    def __init__(self,training_config:TrainingConfig,models:dict,preprocessor:Pipeline,df=None,):
        self.training_config = training_config
        self.logger = self.training_config.logger
        self.dataframe = df
        self.params_path = os.path.join(CONFIGFOLDER,PARAMFILE)
        self.result_path = os.path.join(self.training_config.run_path, "tuning_results_.json")
        self.models = models
        self.params = HyperparameterTuning.read_json_file(self.params_path)
        self.preprocessor = preprocessor

    @staticmethod
    def normalize_class_weight(class_weights):
            if class_weights is None:
                return None

            normalized = []

            for cw in class_weights:
                if cw is None or cw == "balanced":
                    normalized.append(cw)
                elif isinstance(cw, dict):
                    normalized.append({int(k): v for k, v in cw.items()})
                else:
                    raise ValueError(f"Invalid class_weight entry: {cw}")
            return normalized
    
    @staticmethod
    def read_json_file(file_path):
        with open(file_path,'r') as file:
            data = json.load(file)
        
        for params in data.values():
            if isinstance(params, dict) and "class_weight" in params:
                # params['class_weight'] = [HyperparameterTuning.normalize_class_weight(cw) for cw in params['class_weight']]
                params['class_weight'] = HyperparameterTuning.normalize_class_weight(params['class_weight'])
        return data
    
    @staticmethod
    def create_pipeline(preprocessor:Pipeline,model):
        temp_pipeline =[]
        for name,step in preprocessor.steps:
            temp_pipeline.append((name,step))
        temp_pipeline.append(("model",model))
        new_pipeline = Pipeline(temp_pipeline)
        return new_pipeline
    
    @staticmethod
    def create_params(param:dict,type:str):
        if type.lower() =='input':
            final_param = { f"model__{k}":v for k,v in param.items()}
            return final_param
        elif type.lower() =='output':
            final_param = {k.split("__")[-1]:v  for k,v in param.items()}
            return final_param
        else :
            raise ValueError("Value type should be between 'input' & 'output")
        
    def save_mlflow_run(self, model_name, pipe, params:dict,cv_score, x_train,x_test,y_train,y_test):
        experiment_name = f"{self.training_config.run_name}_{self.training_config.timestamp}"
        run_name = f'{model_name}_trial'
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # pipe = self.__class__.create_pipeline(self.preprocessor,model=model)
            pipe.fit(x_train,y_train)
            y_pred = pipe.predict(x_test)
            accuracy = accuracy_score(y_test,y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test,y_pred)    
            mlflow.log_param("params",value=params)
            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("CV score",cv_score)


    def preprocessing_data(self,X:pd.DataFrame,Y:pd.DataFrame):
        x_train,x_test, y_train, y_test  = train_test_split(X,Y,test_size=0.25,random_state=42)
        x_processed,y_processed = self.preprocessor.fit_resample(x_train,y_train)
        # x_train = self.preprocessor.fit_transform(x_train,y_train)
        # x_test = self.preprocessor.transform(x_test)
        return x_processed,x_train,x_test,y_processed,y_train,y_test

    def run_multiple_model_tuning(self, X, Y,
                                scoring='f1', cv=5):
        """
        Runs GridSearchCV for multiple models sequentially and saves results incrementally into a JSON file.
        """

        # Initialize empty JSON file
        x_train,x_test, y_train, y_test  = train_test_split(X,Y,test_size=0.25,random_state=42)
        # x_processed,x_train, x_test,y_processed, y_train, y_test = self.preprocessing_data(X=X,Y=Y)
        with open(self.result_path, "w") as f:
            json.dump({}, f, indent=4)
        print(f"\n📝 Results will be saved in: {self.result_path}")

        results = {}

        for name, grid in self.params.items():
            print(f"\n🔍 Starting hyperparameter tuning for: {name}")

            if name not in self.models:
                print(f"⚠️  No model found for {name}, skipping.")
                continue

            model = self.models[name]
            pipe = self.__class__.create_pipeline(self.preprocessor,model)
            grid = self.__class__.create_params(grid,'input')
            try:
                gs = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=grid,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )

                gs.fit(x_train, y_train)

                # Prepare model results
                best_params = self.__class__.create_params(gs.best_params_,'output')
                model_result = {
                    "best_params": best_params,
                    "best_score": float(gs.best_score_)
                }

                self.save_mlflow_run(name,gs.best_estimator_,best_params,gs.best_score_,x_train,x_test,y_train,y_test)
                # Save in dict
                results[name] = model_result
                
                print(f"✅ Completed: {name}")
                print(f"🏆 Best Score ({scoring}): {gs.best_score_}")
                print(f"🔧 Best Params: {best_params}")

            except Exception as e:
                print(f"❌ Error while tuning {name}")
                print(traceback.format_exc())

                # Save the error also in JSON
                model_result = {"error": traceback.format_exc()}
                results[name] = model_result

            # -----------------------------
            # 🔥 Append results to JSON file
            # -----------------------------
            with open(self.result_path, "r") as f:
                existing_data = json.load(f)

            existing_data[name] = results[name]

            with open(self.result_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            print(f"📌 Saved results for {name} in JSON.")
        return results

if __name__ =="__main__":
    training_config = TrainingConfig()
    df = DataIngestion(training_config=training_config).read_data()
    data_validation = DataValidation(training_config=training_config).initiate_data_validation(df=df)
    if not data_validation:
        print("Code stopping",flush=True)
        sys.exit(1)
    print('Code continuing')
    X,Y = data_validation[1],data_validation[2]
    # print(MODEL_REGISTRY_CLF,flush=True)
    ## Defininf models and pipeline
    main_pipeline = NewAppModel().get_model_pipeline()
    preprocessor_pipeline = get_subpipeline(pipe=main_pipeline)
    # preprocessor_pipeline = get_preprocessor_subpipeline(pipe=main_pipeline)
    best_models = HyperparameterTuning(training_config=training_config,models=MODEL_REGISTRY_CLF,preprocessor=preprocessor_pipeline).run_multiple_model_tuning(X=X,Y=Y)
    print(best_models)
    