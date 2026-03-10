import os
import sys
import pandas as pd 
from src.entity.constants import DATAFOLDER, DATAFILE,SHEETNAME
from src.entity.config_entity import TrainingConfig

##Logging and Exception
# from src.components.logger import logger
from src.components.exception import CustomException

class DataIngestion:
    def __init__(self,training_config:TrainingConfig):
        self.training_config = training_config
        self.data_file = os.path.join(DATAFOLDER,DATAFILE)
        self.logger    = self.training_config.logger
        
    def read_data(self,):
        try:
            if os.path.exists(self.data_file):
                self.logger.info('Data Ingestion started')
                data_type = os.path.splitext(self.data_file)[-1].lower()
                if data_type == '.xlsx':
                    if SHEETNAME is None:
                        raise ValueError("Sheet name is not mentioned in data_config.")
                    df = pd.read_excel(self.data_file,sheet_name=SHEETNAME) 
                elif data_type =='.csv':
                    df = pd.read_csv(self.data_file) 
                else:
                    raise ValueError(f"Unsuported file type : {data_type}")
                self.logger.info('Data Ingestion completed')
                return df 
            else:
                raise ValueError(f"Data file does not exist at : {self.data_file}")
        except Exception as e:
            self.logger.error(f'Error occured : {e}')
            raise CustomException(e,sys)

if __name__ =='__main__':
    training_config = TrainingConfig()
    df = DataIngestion(training_config=training_config).read_data()
    print(df.head(),flush = True) 
