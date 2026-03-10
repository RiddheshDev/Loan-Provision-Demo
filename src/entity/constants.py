from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, SVR
import json
import os

MODEL_REGISTRY_CLF = {    
    "logistic": LogisticRegression(random_state=42),
    "rf": RandomForestClassifier(random_state=42),
    'svc':SVC(random_state=42),
    'xgb':XGBClassifier()
                      }

MODEL_REGISTRY_REG = {
    "logistic" : LinearRegression(),
    "tree" : DecisionTreeRegressor(),
    "rf"   : RandomForestRegressor(random_state=42),
    "svm"  : SVR()
}

ARTIFACT_NAME = 'runs'

## Config & Params
CONFIGFOLDER = 'config'
SCHEMAFILE   = 'schema.json'
PARAMFILE   = 'params.json'

##Validation Report
REPORTFILE = 'report.json'

##Model Saving
MODELCONFIG = "model_param.json"
MODELFOLDER = "model"
MODELREPORT = "report.json"

##Feature score
FEATURESCORETRAIN = "distribution_training_cfg.json"
FEATURESCORELIVE  = "feature_distribution.json"

## DATA CONFIG
if os.path.exists(r"config\data_config.json"):
    with open(r"config\data_config.json",'r',encoding='utf-8') as file:
        data = json.load(file)

required_keys = "datafile"

# Check for missing or invalid values

if "datafile" not in data or data["datafile"] is None:
    raise ValueError(f"datafile is not mentioned in data_config.")

# Safe to assign now
DATAFOLDER = data.get("datafolder","data")
DATAFILE   = data["datafile"]
SHEETNAME  = data.get("sheetname",None)