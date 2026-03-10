from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class InputData(BaseModel):
    full_name : Optional[str] = None
    application_type : Optional[str]
    account_number: Optional[int]
    rent_mortgage_payment: Optional[float] = None
    current_employer: Optional[str] = None
    address: Optional[str] = None
    hire_date: Optional[datetime] = None       # you can change to date if properly formatted
    job_title: Optional[str] = None
    monthly_income: Optional[float] = None
    status: Optional[str] = None
    email: Optional[str] = None
    ln_dti_ratio: Optional[float] = None
    months_employed: Optional[float] = None
    years_employed: Optional[float] = None
    credit_score: Optional[float] = None
    credit_score_date: Optional[datetime] = None   # also convertible to date if needed
    ln_late_1_29: Optional[float] = None
    ln_late_30_59: Optional[float] = None
    ln_late_60_89: Optional[float] = None
    ln_late_90_119: Optional[float] = None
    ln_late_120_149: Optional[float] = None
    ln_late_150_179: Optional[float] = None
    ln_late_180_plus: Optional[float] = None
    neg_balance : Optional[str] = None
    no_deposit : Optional[str] = None
    late_payment : Optional[str] = None
    ln_late_total : Optional[int] = None
    total_months_employed : Optional[int] = None