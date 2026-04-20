# Lab 5B: Trees & Ensembles

## 1. Classification Reports

**Default RF:**
```text
               precision    recall  f1-score   support

           0       0.86      0.98      0.91       753
           1       0.60      0.16      0.26       147

    accuracy                           0.85       900
   macro avg       0.73      0.57      0.59       900
weighted avg       0.82      0.85      0.81       900
```

**Balanced RF:**
```text
               precision    recall  f1-score   support

           0       0.88      0.86      0.87       753
           1       0.36      0.41      0.38       147

    accuracy                           0.78       900
   macro avg       0.62      0.63      0.62       900
weighted avg       0.80      0.78      0.79       900
```


## 2. Top 5 Features by Importance (from RF max_depth=10)
1. `num_support_calls`: 0.267
2. `monthly_charges`: 0.244
3. `total_charges`: 0.185
4. `tenure`: 0.161
5. `contract_months`: 0.084


## 3. PR-AUC Values
- **DT (max_depth=5):** 0.365
- **RF default:** 0.448
- **RF balanced:** 0.419


## 4. ECE Values
- **DT ECE (max_depth=None):** 0.207
- **DT ECE (max_depth=5):** 0.043


## 5. Tree-vs-linear Disagreement
Sample 4060 — tenure=36.0, monthly_charges=20.0, total_charges=1077.33, num_support_calls=2.0, contract_months=1.0, senior_citizen=0.0, has_partner=0.0, has_dependents=0.0.
RF predicts P(churn)=0.60; LR predicts P(churn)=0.17.
The random forest captured the interaction: customers with several support calls on a month-to-month plan are at elevated risk regardless of their relatively long 36-month tenure. Logistic regression can only weight each feature linearly, so it combines the strong long-tenure protection with the individual risks of month-to-month/support calls and arrives at a much lower probability. The tree’s ability to split conditionally on contract_months and then further split on num_support_calls captures a pattern the linear model structurally can’t express.


## 6. Comparison to Week A Logistic Regression
Our decision trees and random forests slightly yield better baseline signals when evaluating performance metrics vs linear combinations. Compared to standard logistic regression, evaluating recall for tree ensembles at the default 0.5 threshold can further optimize detection when appropriately applying `class_weight='balanced'`. This technique effectively boosts recall at the default 0.5 threshold from 0.16 to ~0.41, serving as an effective operating point tool since the actual ranking quality (PR-AUC) remains near identical between default and reweighted approaches.
