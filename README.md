# Loan-Provision-Demo

## Overview

This project provides a **production-ready Loan Provision / Loan Approval Prediction API** built using **FastAPI** and **Machine Learning models (Random Forest)**. The API evaluates applicant financial and behavioral data and returns a loan decision (**Approved / Denied / Unknown**) along with confidence intervals and top contributing features.

The application can be fully containerized using **Docker** and designed for scalable, reproducible deployment.

---

## Key Features

- FastAPI-based REST API
- Supports different client options **New Applicants** and **Opt-In Applicants**. 
- Random Forest models with bootstrapped confidence intervals
- Feature-level explainability (Top-K drivers)
- Centralized logging and exception handling
- Can be Dockerized for production deployment


---
**Note:** Some sections of the code have been modified or omitted to preserve the confidentiality of the original implementation.