import os
import sys
from datetime import datetime
from src.entity.constants import ARTIFACT_NAME
from src.components.logger import CustomLogger

class TrainingConfig:
    def __init__(self,timestamp = datetime.now()):
        self.timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.run_name  = ARTIFACT_NAME
        self.run_path  = os.path.join(self.run_name,self.timestamp)
        os.makedirs(self.run_path)
        self.logger = CustomLogger(timestamp=self.timestamp,run_path=self.run_path).get_logger()
        self.logger.info('TrainingConfig Initialized properly')
        # self.base_dir = os.path.dirname(__file__)
        

