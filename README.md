# Bank Marketing Campaign Classifier Comparison Report

## Jupyter Notebook Link
https://github.com/wjd20170117/Assignment_17_1/blob/main/Assignement17_1/prompt_III.ipynb

## Executive Summary

This report presents a comprehensive analysis of bank marketing campaign data to predict which clients are likely to subscribe to term deposits. Using machine learning classifiers including K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines, we developed models to optimize marketing campaign targeting.

## Key Findings

1. **Best Overall Performance**: Logistic Regression achieved the best test accuracy (90.14%) with good generalization
2. **Overfitting Concerns**: Decision Tree showed significant overfitting (99.54% train vs 83.95% test accuracy)
3. **Computational Efficiency**: KNN was fastest to train (0.30s) while SVM was slowest (193.87s)
4. **Hyperparameter Impact**: Limited improvement from hyperparameter tuning, suggesting good default parameters

## Business Understanding

### Problem Statement
The Portuguese banking institution needs to optimize their telephone marketing campaigns for term deposit subscriptions. Currently, the bank conducts marketing campaigns without sophisticated targeting, leading to inefficient resource allocation and suboptimal conversion rates.

### Business Objective
Develop a predictive model to identify clients most likely to subscribe to term deposits based on:
- Client demographics (age, job, marital status, education)
- Financial status (credit default, housing loan, personal loan status)
- Campaign contact information (contact type, timing)
- Previous campaign outcomes
- Economic indicators

## Modeling Approach

### Baseline Model
- **Strategy**: Most frequent class prediction
- **Baseline Accuracy**: ~88.7%
- **Purpose**: Minimum performance threshold for model evaluation

### Model Selection
Four supervised learning algorithms were evaluated:

1. **Logistic Regression**
   - Linear model with probabilistic interpretation
   - High interpretability for business stakeholders
   - Fast training and prediction

2. **K-Nearest Neighbors (KNN)**
   - Non-parametric, instance-based learning
   - Captures local patterns in data
   - Simple conceptual understanding

3. **Decision Tree**
   - Rule-based model with high interpretability
   - Handles non-linear relationships
   - Can overfit without proper regularization

4. **Support Vector Machine (SVM)**
   - Robust to outliers
   - Effective in high-dimensional spaces
   - Computationally intensive for large datasets

## Model Performance Results

### Initial Model Comparison
| Model | Train Time | Train Accuracy | Test Accuracy |
|-------|------------|----------------|---------------|
| Logistic Regression | 1.1600s | 0.8999 | 0.9014 |
| K-Nearest Neighbors | 0.2998s | 0.9134 | 0.8941 |
| Decision Tree | 2.0390s | 0.9954 | 0.8395 |
| Support Vector Machine | 193.8749s | 0.9048 | 0.9036 |

### Hyperparameter Tuning Results
| Model | Best Parameters | Train Accuracy | Test Accuracy | Tuning Time |
|-------|-----------------|----------------|---------------|-------------|
| Logistic Regression | C=0.1, penalty=l1, solver=liblinear | 0.9003 | 0.9013 | 168.53s |
| K-Nearest Neighbors | n_neighbors=5, weights=uniform | 0.9133 | 0.8938 | 43.01s |
| Decision Tree | max_depth=5, criterion=gini | 0.9027 | 0.9022 | 37.34s |
| Support Vector Machine | C=1, kernel=linear | 0.8975 | 0.8977 | 2806.67s |
