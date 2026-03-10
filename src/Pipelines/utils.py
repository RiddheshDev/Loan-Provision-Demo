import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from typing import Optional
import shap
import bisect
from src.Pipelines.custom_pipeline import NewAppModel, OptInModel
from sklearn.utils import resample
from sklearn.base import clone
 


## Get features names from pipeline
def get_features(model : Pipeline) -> list:
    features = model.named_steps['features selection']
    features = features.get_params().get('kw_args').get('features')
    return features


def get_model(model:Pipeline)-> Optional[Pipeline]:
    estimator = model.named_steps['random forest']
    return estimator

def get_preprocessor_subpipeline(pipe: object) -> Optional[Pipeline]:
    """
    Return a Pipeline containing all steps of `pipe` except the final estimator.
    If `pipe` is not a sklearn.pipeline.Pipeline or has only one step, returns None.

    Usage:
        preproc = get_preprocessor_subpipeline(my_pipeline)
        if preproc is not None:
            # if preproc is already fitted:
            X_trans = preproc.transform(X)
            # or if not fitted:
            X_trans = preproc.fit_transform(X_train)
    """
    if not isinstance(pipe, Pipeline):
        return None

    if len(pipe.steps) <= 1:
        # nothing to extract (pipeline only has estimator)
        return None

    # build a new Pipeline from all but last step
    preproc_steps = pipe.steps[:-2]
    return Pipeline(preproc_steps)

def get_subpipeline(pipe:Pipeline,resampling_enable:bool=True):   
    temp_pipeline = []
    if not hasattr(pipe, "steps"):
        return None
    for name,step in pipe.steps:
        if not resampling_enable and hasattr(step,'fit_resample'):
            continue

        if hasattr(step,'predict'):
            continue

        temp_pipeline.append((name,step))
    return Pipeline(temp_pipeline)

def get_model_pipeline(type:str):
    if type.lower() == "new applicants":
        return NewAppModel().get_model_pipeline()
    
    elif type.lower() == "opt ins":
        return OptInModel().get_model_pipeline()

    else:
        raise ValueError("Pipeline not found")
    
## Getting Weights for features
def get_shap(data,estimator,preprocess:Pipeline,features:list,class_index:int,top_k:int=5):
    preprocess_data = preprocess.transform(data)
    preprocess_data = pd.DataFrame(preprocess_data,columns=features)

    explainer = shap.TreeExplainer(estimator)
    shap_values=explainer.shap_values(preprocess_data)
    shap_vals = shap_values[0,:,class_index]

    shap_df = pd.DataFrame({"features":features,"shap_value": shap_vals,'abs_shap':abs(shap_vals)})
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_k)
    total_abs_shap = shap_df["abs_shap"].sum()
    
    if total_abs_shap == 0:
        shap_df["weight"] = 0.0
    else:
        shap_df["weight"] = shap_df["abs_shap"] / total_abs_shap
    output = {
        row["features"]: round(row["weight"], 2)
        for _, row in shap_df.iterrows()
    }
    return output

### For Confidence Interval
## CI using Bootstrap method
## for training bootstrap models for both application types
def train_bootstrap_models(estimator,preprocess:Pipeline,features:list,x_train,y_train,num_bootstraps:int=100,):
    preprocess_data = preprocess.transform(x_train)
    x_train_processd = pd.DataFrame(preprocess_data,columns=features)
    # bootstrap_predictions = []
    models = []
    for i in range(num_bootstraps):
        # print("No of iterations : ",i,flush=True) ##Only for debugging
        # Step 1: bootstrap sample (same size as original)
        X_sample, y_sample = resample(x_train_processd,y_train)
        # Step 2: train temporary RF model
        temp_model = clone(estimator)
        temp_model.fit(X_sample, y_sample)
        models.append(temp_model)
    return models

## Using in Live for getting the Condifence Intervals.
def bootstrap_ci(preprocess:Pipeline,features:list,model_list:list,input_data,class_index,alpha:float=0.05):
    preprocess_data = preprocess.transform(input_data)
    test_sample = pd.DataFrame(preprocess_data,columns=features)
    prob_list = []
    for model in model_list:
        prob = model.predict_proba(test_sample)[0][class_index]
        prob_list.append(prob)
    ci_lower  = np.percentile(prob_list, 100*(alpha/2))
    ci_upper = np.percentile(prob_list, 100*(1-alpha/2))
    return ci_lower,ci_upper

## Feature Scores
### Code for obtaining feature score dusrin training.
def quantile_method(lower_q:float,upper_q:float,series):
    q_low = series.quantile(lower_q)
    q_high = series.quantile(upper_q)
    iqr = q_high-q_low
    lower_fence = q_low -  1.5*iqr 
    upper_fence = q_high + 1.5*iqr
    capped_series = series.clip(lower=lower_fence,upper=upper_fence)
    return capped_series

def compute_feature_distributions(df, feature_config):
    dist = {}
    for feature, cfg in feature_config.items():
        # print(feature)
        if cfg["type"] == "numeric":
            lower_q = cfg.get("iqr_lower", 0.25)
            upper_q = cfg.get("iqr_upper", 0.90)
            capped_series = quantile_method(lower_q,upper_q,df[feature].dropna())
            qs = np.linspace(0, 1, cfg["quantiles"] + 1)
            quantile_values = np.quantile(capped_series.dropna(), qs)

            dist[feature] = {
                "type": "numeric",
                "direction": cfg["direction"],
                "quantiles": {
                    str(round(q, 2)): round(float(v),2)
                    for q, v in zip(qs, quantile_values)
                }
            }

        elif cfg["type"] == "binary":
            counts = df[feature].value_counts(normalize=True)

            dist[feature] = {
                "type": "binary",
                "good_value": cfg["good_value"],
                "scores": {
                    str(k): round(v * 100)
                    for k, v in counts.items()
                }
            }
    return dist

### Calculate feature score in Production.
def numeric_feature_score(x, dist):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 50
    quantiles = dist["quantiles"]
    direction = dist["direction"]

    items = sorted((float(q), v) for q, v in quantiles.items())
    qs = [q for q, _ in items]
    values = [v for _, v in items]
    # print(x,flush=True)
    if x <= values[0]:
        percentile = 0
    elif x >= values[-1]:
        percentile = 100
    else:
        idx = bisect.bisect_left(values, x)
        x0, x1 = values[idx - 1], values[idx]
        q0, q1 = qs[idx - 1] * 100, qs[idx] * 100
        if x1 == x0:
            percentile = q1
        else:
            percentile = q0 + (x - x0) * (q1 - q0) / (x1 - x0)

    if direction == "higher_is_better":
        return round(percentile)
    else:
        return round(100 - percentile)

def binary_feature_score(x, dist):
    scores = dist["scores"]
    return scores.get(str(x), 50)

def feature_score(feature, value, feature_dists,default_score = 50):
    dist = feature_dists.get(feature)
    if dist is None:
        return default_score
    if dist["type"] == "numeric":
        return numeric_feature_score(value, dist)

    if dist["type"] == "binary":
        return binary_feature_score(value, dist)
    return 50

def get_feature_score(top_k:dict,input_value,feature_dists:dict):
    results = {}

    for feature,shap_value in top_k.items():
        if feature not in input_value:
            score = 50
        # print("feature",k,"input_value",input_value[k].iloc[0],flush=True)
        else:
            score = feature_score(feature=feature,value=input_value[feature].iloc[0],feature_dists=feature_dists)
        results[feature] = {'weights':shap_value,
                            'score':score
                            }
    return results

## Getting all output together.
def final_output(data:pd.DataFrame,estimator,model_list,output,feature_dist,threshold:float=0.75):
    accept_prob = output[1]
    reject_prob = output[0]
    # prob_diff = abs(accept_prob-reject_prob)
    model = get_model(estimator)
    preprocess = get_preprocessor_subpipeline(estimator)
    features = get_features(estimator)
    data['total_months_employed'] = data['months_employed'].fillna(0) + data['years_employed'].fillna(0)*12
    if accept_prob >=threshold:
        prediction = "Approved"
        prob = round(float(accept_prob),3)
        # ci_lower, ci_upper = model_based_ci(model,preprocess,data,features,1,0.05)
        ci_lower, ci_upper = bootstrap_ci(preprocess,features,model_list,data,class_index=1)
        ci_upper = max(ci_upper,prob)
        topk_list = get_shap(data=data,preprocess=preprocess,estimator=model,features=features,class_index=1,top_k=5)
        topk_list = get_feature_score(topk_list,data,feature_dists=feature_dist)

    elif reject_prob >= threshold:
        prediction = "Denied"
        prob = round(float(reject_prob),3)
        # ci_lower, ci_upper = model_based_ci(model,preprocess,data,features,0,0.05)
        ci_lower, ci_upper = bootstrap_ci(preprocess,features,model_list,data,class_index=0)
        ci_upper = max(ci_upper,prob)
        topk_list = get_shap(data=data,preprocess=preprocess,estimator=model,features=features,class_index=0,top_k=5)
        topk_list = get_feature_score(topk_list,data,feature_dists=feature_dist)
    
    else : 
        prediction = 'Unknown'
        prob = max(accept_prob, reject_prob)
        ci_lower , ci_upper = bootstrap_ci(preprocess,features,model_list,data,class_index=1)
        ci_upper = max(ci_upper,prob)
        topk_list = get_shap(data=data,preprocess=preprocess,estimator=model,features=features,class_index=1,top_k=5)
        topk_list = get_feature_score(topk_list,data,feature_dists=feature_dist)
    row_out = {
        "prediction": prediction,
        "prediction_score": round(float(prob), 3),
        "ci_lower": round(float(ci_lower), 3) if ci_lower is not None else None,
        "ci_upper": round(float(ci_upper), 3) if ci_upper is not None else None,
        "topk_list": topk_list if topk_list is not None else None
    }
    return row_out