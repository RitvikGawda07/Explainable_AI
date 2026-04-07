# Explainable_AI

A toxic comment classification project that compares **LIME explanations** and **counterfactual text explanations** for **faithfulness** and **actionability** on a YouTube comments toxicity dataset. The project trains a **Logistic Regression** classifier on **TF-IDF** text features, then uses explainability methods to show why a comment is predicted as toxic and how it could be changed to flip the prediction. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Project Goal

This project explores the research question:

**For a toxic comment classifier on the YouTube comments dataset, how do LIME explanations and counterfactual text explanations compare in terms of faithfulness and actionability?** :contentReference[oaicite:2]{index=2}

The motivation is to support **content moderation** by helping:
- **Moderators** understand why a comment was flagged
- **Users** understand what caused the toxic prediction
- **Users** make realistic edits to comments to avoid toxic classifications 

## Dataset

The project uses the **YouTube toxic comments dataset** (`youtoxic_english_1000.csv`) with text comments and binary toxicity labels. In the code, the dataset columns are standardized to:
- `text`
- `label` 

## Methods

### 1. Toxic Comment Classifier
A **Logistic Regression** model is trained using **TF-IDF** vectorized comment text. The data is split into training and test sets before model fitting. 

### 2. LIME Explanations
**LIME (Local Interpretable Model-agnostic Explanations)** is used to explain individual predictions by assigning importance weights to words in a comment. Words with higher positive weights push the prediction toward the **toxic** class, while negative weights push it away. 

### 3. Counterfactual Explanations
A simple counterfactual method is implemented by:
- generating LIME explanations
- identifying words contributing most to toxicity
- removing those words one by one
- checking whether the prediction flips from **toxic** to **non-toxic** 

## Evaluation Criteria

The explanations are compared on two dimensions:

### Faithfulness
Evaluated using a **deletion test**, where removing the most important words should change the model prediction more than removing random words. This is mainly used to assess **LIME**. 

### Actionability
Evaluated by whether the counterfactual explanation suggests **small, realistic edits** that flip a prediction from toxic to non-toxic. This is mainly used to assess **counterfactual explanations**. 

## Key Findings

- **LIME** is useful for showing which words most strongly influence a prediction.
- **Counterfactual explanations** are more actionable because they suggest how to modify a comment to change the model’s decision.
- The classifier relies heavily on **keyword-based signals**, which is expected for a TF-IDF + Logistic Regression setup.
  
## Limitations

### LIME
- Focuses on individual words rather than full sentence meaning
- Cannot capture word order with TF-IDF features
- May over-rely on swear words or isolated tokens

### Counterfactuals
- The current implementation is a simple baseline
- Removing one word can produce grammatically incorrect text
- A prediction may flip even if the edited sentence still sounds toxic
- The method depends heavily on LIME stability 

## Repository Structure

```text
.
├── EAI_final_assignment_code.py      # Main Python implementation
├── youtoxic_english_1000.csv         # Dataset
├── EAI_Final_Assignment.docx         # Final report
├── Intermediate_report.pdf           # Intermediate report
└── README.md

pip install pandas scikit-learn lime matplotlib

