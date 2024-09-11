# Content Moderatio and Toxicity Classification
## Introduction

This project focuses on developing models for content moderation and toxicity classification, aiming to accurately identify and flag harmful content based on predefined metrics. The dataset used consists of training and test data with gold labels, containing around 159.5K training set comments and 153K test data comments, categorized into six classes: toxic, severe toxic, obscene, threat, insult, and identity hate. The dataset is notably imbalanced, with a significant portion favoring the neutral class, which poses a challenge for accurate classification.

## Dataset Overview

The training dataset consists of 159,571 comments distributed across seven classes. The data distribution is highly imbalanced, skewed towards the toxic class. A notable portion of the dataset belongs to the neutral class, which is not portrayed in the distribution graph. This imbalance affects model performance; for instance, assigning the label '0' to the entire dataset would result in nearly 90% accuracy.

To address this, we created a balanced dataset by splitting the data into two categories: toxic (any label is '1') and clean (all labels are '0'). We sampled 15,000 instances from each category to form a balanced dataset of 30,000 instances for training, which improved model performance significantly.

![Unknown](https://github.com/user-attachments/assets/dc572248-9a98-490b-acc8-5b701bb899e5)

## Model Implementations

### 1. Logistic Regression Model
#### Data Preprocessing:
The preprocessing steps included text cleaning (removal of URLs, punctuation, unnecessary spaces, special characters, and stop words) and text normalization (lowercasing, lemmatization, and stemming). After cleaning, frequency-based preprocessing of top tokens was performed, and labels were assigned to both test and train data.

#### Model Details:
We used a one-vs-rest classifier with logistic regression from the scikit-learn library. The model was trained for 1000 iterations, predicting test data and calculating accuracy, ROC AUC, and F1 score. Results demonstrated 89.96% accuracy, a 70.55% ROC AUC score, and a 62.75% F1 score. The model performed well in predicting non-toxic comments but struggled with severe toxic, threat, and identity hate categories due to lower true positive and higher false negative values.

#### Limitations:
One-vs-rest classification using logistic regression is suboptimal for multi-label text classification due to its independent assumption between labels and oversimplification of the complex interactions in natural language. Neural network-based approaches like BERT are more suitable for this problem.

![Snip 2024-09-11 17 41 47](https://github.com/user-attachments/assets/2cfaff7b-2c51-47f9-8a77-99abd9b769e2)

### 2. LSTM Model
Data Preprocessing:
The preprocessing steps were similar to those in the logistic regression model. The process included tokenization and padding using a Keras Tokenizer with a vocabulary size of 5000 to reduce computational load. The tokenizer transformed comments into integer sequences, and padding/truncation was applied to ensure consistent input length for the LSTM model.

#### Model Architecture:
The LSTM model comprised an Embedding layer, an LSTM layer, and a Dense layer, optimized for sequential data processing for toxicity classification. The architecture was designed to transform word indices into dense vectors mapped to six toxicity categories. Key parameters included a vocabulary size of 5000, an embedding dimension of 100, a hidden dimension of 256, and an output dimension of 6. The model used Binary Cross-Entropy with Logits Loss and the Adam optimizer with a learning rate of 0.001.

#### Performance and Challenges:
The LSTM model showed proficiency in some categories like 'toxic' and 'obscene' but struggled with 'threat' and 'identity hate' due to class imbalance. Strategies like oversampling, undersampling, and advanced preprocessing techniques offer potential for improvement.

### 3. BERT Model
#### Model Implementation:
We used a DistilBERT-based architecture for multi-label text classification. The tokenization process involved a custom class applying a tokenizer to the dataset, adding unique tokens like [CLS], [SEP], and [PAD], and creating an attention mask for variable-length sequences. The tokenized and padded data were fed into a DistilBERT model for training and evaluation.

## Model Details:
The model utilized AdamW as the optimizer with a weight decay term to mitigate overfitting. No specific loss function was used; instead, the loss returned by the model was used in the backward pass. The architecture included 6 hidden layers, 12 attention heads, and a hidden size of 768. Training was conducted for three epochs with a learning rate of 2e-5 and a dropout probability of 0.2.

## Results:
The BERT model achieved an accuracy of 87%. The evaluation metrics included the ROC curve, Area Under the Curve (AUC), accuracy, and a classification report, demonstrating promising results in handling multi-label text classification for toxicity.
![Snip 2024-09-11 17 42 50](https://github.com/user-attachments/assets/10d138f2-69e1-4c2f-9f4c-b2dc3291aad0)

![Unknown-2](https://github.com/user-attachments/assets/526f1e9f-f05d-4bb4-aa4d-3174fb173f94)


## Summary

The project aimed to enhance content moderation through toxicity classification. The dataset's imbalance, favoring the neutral class, posed challenges. We created a balanced dataset to address this. Logistic Regression excelled in predicting non-toxic comments but struggled with severe toxicity, threats, and identity hate. The LSTM model faced similar challenges, prompting the exploration of more advanced models. The DistilBERT implementation showcased promising results with 87% accuracy, highlighting the need for advanced approaches like BERT for effective content moderation.

## Future Work

### Further improvements can be made by:

Experimenting with more advanced neural network architectures.
Exploring different techniques for handling class imbalance, such as SMOTE or class-weighted loss functions.
Using ensemble methods to combine the strengths of different models.
