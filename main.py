import os 
import sys
from datetime import datetime
from src.entity.config_entity import TrainingConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
# from src.Pipelines.custom_pipeline import NewAppModel, OptInModel
from src.components.hyperparameter_tuning import  HyperparameterTuning
from src.components.training import ModelTraining
from src.entity.constants import MODEL_REGISTRY_CLF,MODEL_REGISTRY_REG,SHEETNAME
from src.Pipelines.utils import get_subpipeline, get_model_pipeline
import argparse


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

    main_pipeline = get_model_pipeline(SHEETNAME)
    preprocessor_pipeline = get_subpipeline(pipe=main_pipeline)
    best_models=None
    if args.tuning:
        print('Parameter tuning')
        best_models = HyperparameterTuning(training_config=training_config,models=MODEL_REGISTRY_CLF,preprocessor=preprocessor_pipeline).run_multiple_model_tuning(X=X,Y=Y)
        # print(best_models)
    print('Training Model')
    results = ModelTraining(training_config,preprocessor_pipeline,MODEL_REGISTRY_CLF,best_models).train_model(X,Y,extra_output=True)
    print(results)
    