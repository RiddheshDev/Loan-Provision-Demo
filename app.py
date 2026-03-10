## Required Imports for FastAPi
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request
import uvicorn 

## Imports for Models
import pandas as pd
import cloudpickle
import dill
import json
## Imports required for logging & Exception
from src.components.logger import logger
from src.components.exception import CustomException
from src.components.data_model import InputData
from src.Pipelines.utils import final_output
import sys

import warnings
warnings.filterwarnings('ignore')

with open(r"src/models/new_app_model_forrest_v2.pkl",'rb') as file:
    model1 = cloudpickle.load(file)
    print("Model1 loaded succesfully",flush=True)

with open(r"src/models/opt_in_model_forrest_v2.pkl",'rb') as file:
    model2 = cloudpickle.load(file)
    print("Model2 loaded succesfully",flush=True)

with open(r'src/models/bootstrap_models_new_app_v2.pkl','rb') as file:
    models_list_new_app = dill.load(file)

with open(r'src/models/bootstrap_models_opt_in_v2.pkl','rb') as file:
    models_list_opt_in = dill.load(file)

with open(r"src/models/feature_distributions_v2.json") as f:
    FEATURE_DISTS = json.load(f)

app = FastAPI(title='Loan Approval')

@app.get("/")
def home():
    return {'message':"Welcome to the Loan approval page"}

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post('/predict')
def predict(data : InputData):
    print("============================================================================================",flush=True)
    print('Data received',[data.model_dump()],flush=True)
    try:
        logger.info('Data received')
        application_type = str(data.model_dump().get('application_type','')).lower().strip()
        if application_type =='new applicants':
            print('New Applicants branch',flush=True)
            input_df = pd.DataFrame([data.model_dump()])
            print(input_df.head(),flush=True)
            # Get prediction
            # prediction = model1.predict(input_df)[0]
            output = model1.predict_proba(input_df)[0]
            result = final_output(input_df,model1,models_list_new_app,output,feature_dist=FEATURE_DISTS[application_type])
            print('Prediction done',flush=True)

        elif application_type =='opt in':
            print('Opt Ins branch',flush=True)
            input_df = pd.DataFrame([data.model_dump()])
            print(input_df.head(),flush=True)
            # Get prediction
            # prediction = model2.predict(input_df)[0]
            output = model2.predict_proba(input_df)[0]
            result = final_output(input_df,model2,models_list_opt_in,output,feature_dist=FEATURE_DISTS[application_type])
        else :
            return {"error": "Invalid or missing application_type. Expected 'new applicants' or 'opt in'."}
    
        return result
    
    except Exception as e:
        logger.error(f'Error occured : {e}')
        raise CustomException(e,sys)

@app.exception_handler(Exception)
async def generic_exeception_handler(request:Request,exc:Exception):
    logger.error(f"Unexpected error : {exc}")
    return JSONResponse(status_code=500,
                        content={"error":"Internal server error","detials":str(exc)}
                        )

# if __name__ =="__main__":
#     uvicorn.run("app:app",host="0.0.0.0",port=5050,reload=True)
