# Predictive Maintenance Dashboard

An explainable AI system for industrial machinery fault detection and 
predictive maintenance, built on the AI4I 2020 Predictive Maintenance 
dataset (UCI Machine Learning Repository).

## Overview

Instead of just classifying whether a machine will fail, this project 
turns a prediction into a full decision-support tool a maintenance team 
can actually act on — telling them not just *what* is wrong, but *why*, 
*how urgent* it is, *what it will cost* if ignored, and *how risk is 
likely to evolve* over the next 24 hours.

## Key Features

- **Dual-head neural network** — a single model that simultaneously 
  predicts (1) failure risk and (2) failure type (Tool Wear, Heat 
  Dissipation, Power, Overstrain, or Random Failure)
- **SHAP explainability** — every prediction comes with a feature-level 
  breakdown of what drove the risk score
- **Self-baseline anomaly detection** — flags readings against a 
  machine's own historical norms rather than one global threshold
- **24-hour risk forecast** — projects how risk may evolve based on 
  current stress factors
- **Cost-of-inaction estimator** — converts predicted severity into an 
  estimated downtime and financial impact
- **Three interaction modes** — manual single-machine input, batch CSV 
  upload for fleet-wide analysis, and a live "what-if" simulator
- Achieves **ROC-AUC of 0.963** on held-out test data, with SMOTE-based 
  handling of the dataset's severe class imbalance (~3.4% failure rate)

## Tech Stack

- **Backend:** Python, Flask, TensorFlow/Keras, Scikit-learn, 
  Imbalanced-learn (SMOTE), SHAP
- **Frontend:** HTML5, CSS, JavaScript, Chart.js
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)

## Why This Project

Most predictive maintenance research stops at reporting accuracy metrics 
on a benchmark dataset. This project wraps a correctly-scoped model in a 
complete decision layer — explainability, anomaly context, forecasting, 
and cost translation — so the output is something a maintenance engineer 
could actually use, not just a number in a notebook.
