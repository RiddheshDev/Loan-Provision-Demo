import os
import pandas as pd
import sys
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingConfig
from src.components.exception import CustomException

from src.entity.constants import CONFIGFOLDER,SCHEMAFILE,REPORTFILE,SHEETNAME
import yaml
import json

class DataValidation:
    def __init__(self,training_config:TrainingConfig):
        self.training_config = training_config
        self.logger = self.training_config.logger
        self.config_path = os.path.join(CONFIGFOLDER,SCHEMAFILE)
        self.data_schema = self.__class__.read_file(self.config_path)
        self.features = self.data_schema['features']
        self.Y  = self.data_schema['dependent']

    # @staticmethod    
    # def read_yaml(file_path):
    #     with open(file_path,'rb') as file:
    #         data = yaml.safe_load(file)
    #         # print('Data',data,flush=True)  ## For debugging
    #     return data
    @staticmethod
    def read_file(file_path) -> dict:
        data_type = os.path.splitext(file_path)[-1]
        sheet_name = SHEETNAME.lower()
        if data_type == ".json":
            with open(file_path,'r',encoding="utf-8") as file:
                data =json.load(file)
                return data[sheet_name]
        elif data_type == ".yaml":
            with open(file_path,'rb') as file:
                data = yaml.safe_load(file)
            # print('Data',data,flush=True)  ## For debugging
                return data[sheet_name]  
        else:
            raise ValueError("Required file format not found for schema.")
        
    def column_check(self,df:pd.DataFrame):
        try:
            self.logger.info('Checking columns')
            missing_column = [col for col in self.features if col not in df.columns]
            # print('Missing_columns : ',missing_column)
            if missing_column:  
                self.logger.critical(f'Missing columns : {missing_column}')
                return True , missing_column
            return False 
        
        except Exception as e:
            self.logger.error(f"Error occured : {e}")
            raise CustomException(e,sys)

    def shape_check(self,df:pd.DataFrame,threshold:int=1):
        try:
            self.logger.info("Checking Shape")
            if df.shape[0] >=threshold:
                return False
            self.logger.critical(f'Improper Shape : {df.shape[0]}')
            return True
        
        except Exception as e:
            self.logger.error(f"Error occured : {e}")
            raise CustomException(e,sys)
            
    def save_report(self,report):
        save_path = os.path.join(self.training_config.run_path,REPORTFILE)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

    def initiate_data_validation(self,df):
        df.columns = ["_".join(col.replace('+',' plus').replace('/','_').replace('-','_').split()) for col in df.columns]
        column_status= self.column_check(df=df)
        shape_status = self.shape_check(df=df,threshold=1)
        report = {}
        if column_status : 
            report['column_status'] = {'status':'Fail','missing_colum':column_status[1]}
        if shape_status : 
            report['shape_status'] = {'status':'Fail'}
        if report:
            self.save_report(report=report)
            validation_status = False
            return validation_status
        validation_status = True
        X = df.loc[:,df.columns != "status"]   ## Gets all the features for further processing
        # X = df.loc[self.features]            ## Gets only selected features for processing.
        Y = df[self.Y]
        return validation_status, X, Y

if __name__ == '__main__':
    training_config = TrainingConfig()
    df = DataIngestion(training_config=training_config).read_data()
    data_validation = DataValidation(training_config=training_config).initiate_data_validation(df=df)
    if not data_validation:
        print("Code stopping",flush=True)
        sys.exit(1)
    print('Code continuing')
    X,Y = data_validation[1],data_validation[2]
    print(X.head(),flush=True)
    print(Y.head(),flush=True)
    