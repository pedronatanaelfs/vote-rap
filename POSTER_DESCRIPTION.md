# Poster Description: Vote Outcome Prediction using Temporal Evidence of Related Approval Patterns

This document provides a detailed explanation of the concepts, methodology, and results presented in the poster *Vote Outcome Prediction using Temporal Evidence of Related Approval Patterns*, authored by Pedro N. F. da Silva and colleagues, and presented at the **2025 Doctoral Consortium School – ADBIS**, Tampere University.

---

## 1. Context and Motivation

The Brazilian Chamber of Deputies consists of 513 members, with highly dynamic and complex alliances influenced by multiparty politics, ideologies, and shifting coalitions. Understanding legislative behavior in this environment requires computational tools capable of analyzing and predicting voting patterns.

The increasing availability of **open legislative data** from the Brazilian Chamber — including roll‑call votes, proposition metadata, deputies, parties, and sessions — enables researchers to model political behavior.  
The poster proposes a machine‑learning approach to predict whether a **proposition (bill)** will be **approved or rejected**, leveraging **temporal evidence** about how similar proposals and parties performed in the past.

---

## 2. Data Collection and Preparation

The authors build a proposition‑level dataset using the official open data portals of the Chamber of Deputies. Each row represents a proposition that went through a nominal roll‑call vote.

### Sources used include:
- Roll‑call vote records  
- Proposition metadata  
- Deputies and party information  
- Legislature and session details  

### Data preparation steps:
- Cleaning and harmonizing identifiers  
- Keeping only propositions with clear “approved” or “rejected” outcomes  
- Ensuring no temporal leakage: **all features for a proposition use only past data**  
- Chronological split into **80% training** and **20% testing**

---

## 3. Feature Engineering

A central contribution of the poster is the design of **three temporal and structural features**:

### 3.1. Vote Orientation  
Represents the coalition/ideological stance associated with each proposition, capturing how parties position themselves (government vs. opposition, left vs. right).

### 3.2. Party Popularity  
A party‑level metric indicating how successful a party has been at getting its authored propositions approved over a chosen time window.  
High popularity = the party’s proposals are frequently approved.

### 3.3. Historical Approval Rate (HAR)  
A feature representing the recent empirical probability that **similar propositions** were approved within time windows of 1, 2, 3, 4, 5, or 10 years.  
HAR reflects institutional memory and long‑term tendencies.

These features capture more than ideological polarity—they encode structural, historical, and temporal signals that strongly influence legislative outcomes.

---

## 4. Modeling Approach

The prediction model is based on an **XGBoostClassifier**, chosen for its robustness and ability to learn nonlinear relationships.

### Modeling details:
- **Two‑stage hyperparameter optimization**
  1. *RandomizedSearchCV* for wide exploration  
  2. *GridSearchCV* for fine‑tuning  
- Evaluation metric for both search stages: **AUROC**  
- **3‑fold Stratified K‑Fold** for cross‑validation  
- **StandardScaler** applied to numeric features  
- Final feature set:
  - vote_orientation  
  - party_popularity  
  - historical_approval_rate

---

## 5. Results and Analysis

The model, called **VOTE‑RAP**, significantly improves predictive performance compared to a baseline model.

### Key improvements:
- **AUROC**:  
  - Baseline: 0.8599  
  - VOTE‑RAP: 0.9108  
  - **+5.9 percentage points**

- **F1‑score for rejected class**:  
  - Baseline: 0.637  
  - VOTE‑RAP: 0.700  
  - **+9.9 percentage points**

Additional evaluation includes:
- Threshold tuning to maximize *F1_rejected*  
- Detailed confusion matrix and PR curves  
- Temporal analysis showing that performance varies with political stability:
  - High polarization → better predictions  
  - Political transitions/crises → temporarily worse performance

---

## 6. Limitations and Future Work

- Performance decreases during periods of political instability  
- Future work may incorporate:
  - More advanced temporal models  
  - Richer contextual features  
  - Complex network‑based metrics

---

## 7. Related Work

VOTE‑RAP builds upon two main research areas:

1. **Roll‑call prediction with embeddings and neural models**  
   (e.g., Patil et al., Kraft et al.)

2. **Complex network approaches to legislative behavior**  
   (e.g., Brito, Silva & Amancio; Ferreira et al.; Cherepnalkoski et al.)

The novelty lies in explicitly modeling **temporal approval patterns** as structured features in a supervised classifier.

---

## 8. Summary

The poster introduces:
- A cleaned, temporal, proposition‑level dataset  
- Three engineered features capturing ideological, structural, and temporal evidence  
- A tuned XGBoost model that substantially improves prediction accuracy  
- Strong results, especially in predicting **rejected propositions**, which are harder to classify  
- Insightful analysis on how political environment affects model performance

This work offers a transparent, interpretable, feature‑based approach to understanding legislative outcomes in highly dynamic political systems like Brazil’s Chamber of Deputies.

