from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from src.Pipelines.column_transformers import *
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.components.logger import logger
from src.components.exception import CustomException

@dataclass
class FeaturesList:
    ## New Applicants
    cat_columns_new_app = ['current_employer', 'address', 'job_title', 'email']
    binary_columns_new_app = ['neg_balance', 'no_deposit', 'late_payment']
    personal_finance_new_app = ['ln_dti_ratio', 'months_employed', 'years_employed', 'credit_score']
    ln_late_cols_new_app = [ 'ln_late_1_29','ln_late_30_59','ln_late_60_89','ln_late_90_119','ln_late_120_149','ln_late_150_179','ln_late_180_plus']
    log_columns_new_app = ['rent_mortgage_payment', 'monthly_income', 'ln_dti_ratio', 'credit_score']
    imputing_cols_new_app = [ 'rent_mortgage_payment', 'monthly_income','ln_dti_ratio','credit_score', 
                        'ln_late_1_29','ln_late_30_59','ln_late_60_89','ln_late_90_119','ln_late_120_149','ln_late_150_179','ln_late_180_plus']
    features_new_app = [
                        'rent_mortgage_payment', 'monthly_income','ln_dti_ratio','credit_score',  'total_months_employed', 
                        'ln_late_1_29','ln_late_30_59','ln_late_60_89','ln_late_90_119','ln_late_120_149','ln_late_150_179','ln_late_180_plus',
                        'neg_balance', 'no_deposit', 'late_payment',
                        ]
    ## Opt Ins
    cat_columns_opt_in = ['email']
    binary_columns_opt_in = ['neg_balance', 'no_deposit', 'late_payment']
    personal_finance_opt_in = ['ln_dti_ratio', 'months_employed', 'years_employed', 'credit_score']
    ln_late_cols_opt_in = [ 'ln_late_1_29','ln_late_30_59','ln_late_60_89','ln_late_90_119','ln_late_120_149','ln_late_150_179','ln_late_180_plus']
    log_columns_opt_in = ['monthly_income', 'ln_dti_ratio', 'credit_score']
    imputing_cols_opt_in = ['monthly_income','ln_dti_ratio','credit_score', 
                        'ln_late_1_29','ln_late_30_59','ln_late_60_89','ln_late_90_119','ln_late_120_149','ln_late_150_179','ln_late_180_plus']
    features__opt_in= [
                'monthly_income','ln_dti_ratio','credit_score',  'total_months_employed', 
                'ln_late_1_29','ln_late_30_59','ln_late_60_89','ln_late_90_119','ln_late_120_149','ln_late_150_179','ln_late_180_plus',
                'neg_balance', 'no_deposit', 'late_payment',
                ]

class NewAppModel:
    def __init__(self):
        ##Features list importing
        self.constants = FeaturesList()
        
        ##Sub-Pipelines
        self.cat_columns_pipeline_new_app = Pipeline(steps=[('cat_columns',SimpleImputer(strategy='most_frequent'))])
        self.binary_encoding_pipeline_new_app = Pipeline(steps=[('binary_encoding',FunctionTransformer(binary_encoding,kw_args={'cols':self.constants.binary_columns_new_app}))])
        self.imputing_pipeline_new_app = Pipeline(steps=[('imputing all columns',IterativeImputer(random_state=42,max_iter=10)),
                                                         ('round_off',FunctionTransformer(round_off,kw_args={'cols':self.constants.ln_late_cols_opt_in}))])

        ##Column Transformer
        self.log_processor_new_app = ColumnTransformer(transformers=[('log transformation',self.log_transformation_pipeline_new_app,self.constants.log_columns_new_app)
                                                        ],remainder='passthrough'
                                                            ).set_output(transform="pandas")   
        
    def get_model_pipeline(self):
        ##Final Model Pipeline
        pipeline = Pipeline(steps=[
                               ('rename columns1',FunctionTransformer(rename_column)),
                               ('Add new columns',TotalMonthsEMP()),
                                ('outlier treatment',CreditScoreOutlier()),
                                ('features selection',FunctionTransformer(selected_features,kw_args={'features':self.constants.features_new_app})),
                                ('missing values',SimpleImputer(strategy='median')),
                               ('scaling variables',StandardScaler()),
                               ('smote',SMOTETomek(sampling_strategy='auto',random_state=42)),
                            #    ('logistic',LogisticRegression(max_iter=1000,random_state=42)),
                               ('random forest',RandomForestClassifier(**{'n_estimators': 200,
                                                                        'min_samples_split': 5,
                                                                        'min_samples_leaf': 1,
                                                                        'max_features': 'log2',
                                                                        'max_depth': None,
                                                                        'class_weight': None}))
                                ])
        return pipeline
    
class OptInModel:
    def __init__(self):
        ##Features list importing
        self.constants = FeaturesList()

        ##Sub-Pipelines
        self.cat_columns_pipeline_opt_in = Pipeline(steps=[('cat_columns',SimpleImputer(strategy='most_frequent'))])
        self.binary_encoding_pipeline_opt_in = Pipeline(steps=[('binary_encoding',FunctionTransformer(binary_encoding,kw_args={'cols':self.constants.binary_columns_opt_in}))])
        self.imputing_pipeline_opt_in = Pipeline(steps=[('imputing all columns',IterativeImputer(random_state=42,max_iter=10)),
                                                        ('round_off',FunctionTransformer(round_off,kw_args={'cols':self.constants.ln_late_cols_opt_in}))])
        
        ##Column Transformers
        self.log_processor_opt_in = ColumnTransformer(transformers=[('log transformation',self.log_transformation_pipeline_opt_in,self.constants.log_columns_opt_in)
                                                        ],remainder='passthrough'
                                                            ).set_output(transform="pandas")
        
    def get_model_pipeline(self):
        ##Final Model Pipeline
        pipeline = Pipeline(steps=[('Impute columns',self.columns_preprocessor_opt_in),
                               ('rename columns1',FunctionTransformer(rename_column)),
                               ('Add new columns',TotalMonthsEMP()),
                                ('outlier treatment',CreditScoreOutlier()),
                                ('log transformation',self.log_processor_opt_in),
                                ('rename columns2',FunctionTransformer(rename_column)),
                               ('scaling variables',StandardScaler()),
                               ('smote',SMOTETomek(sampling_strategy='auto',random_state=42)),
                               ('random forest',RandomForestClassifier(n_estimators=100,    # number of trees
                                max_depth=None,      # allow trees to expand fully
                                random_state=42,
                                n_jobs=-1))
                                ])
        return pipeline

