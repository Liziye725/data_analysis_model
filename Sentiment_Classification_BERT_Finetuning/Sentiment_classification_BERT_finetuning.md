# BERT-Based Text Classification

This repository contains code for a text classification task using a BERT-based neural network model. The goal of this project is to classify sentences into two distinct categories using natural language processing (NLP) techniques with a pre-trained BERT model from Hugging Face's `transformers` library.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model Parameters and Choices](#model-parameters-and-choices)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Reference](#reference)

## Overview

The primary objective of this project is to classify sentences into two classes using a pre-trained BERT model fine-tuned on the dataset. The BERT (Bidirectional Encoder Representations from Transformers) model is chosen due to its state-of-the-art performance on various NLP tasks.

## Dataset

The dataset used in this project consists of sentences and their corresponding labels, which are categorized into two classes (0 and 1). The dataset was read using `pandas` and explored to understand its distribution and characteristics.

## Model Architecture

The model architecture is based on the BERT model, specifically the "bert-base-uncased" variant from Hugging Face's `transformers` library. BERT is a transformer-based model that uses self-attention mechanisms to provide contextual embeddings for each word in a sentence.

### Custom BERT Model

We extended the BERT model by adding a few custom layers for fine-tuning:

1. **BERT Encoder**: We use a pre-trained BERT model (`bert-base-uncased`) to encode the input sentences into contextual embeddings. The BERT model outputs a hidden state for each token, with the `[CLS]` token representing the entire sentence's embedding.
2. **Fully Connected Layers**: Two fully connected layers (`fc1` and `fc2`) are added on top of BERT:

   - `fc1`: A dense layer with 512 neurons that reduces the 768-dimensional output from BERT.
   - `fc2`: A dense layer with 2 neurons, representing the two classes for the classification task.

3. **Dropout Layer**: A dropout layer with a probability of 0.2 is used for regularization to prevent overfitting by randomly zeroing out some of the activations during training.

4. **Activation Functions**:
   - **ReLU (Rectified Linear Unit)**: Used after `fc1` to introduce non-linearity and help the model learn more complex patterns.
   - **Log Softmax**: Used in the final layer to convert the raw scores into log-probabilities for the two classes. This output is suitable for the negative log-likelihood loss function (`nn.NLLLoss`), which is used during training.

### Why This Model and Parameters?

- **BERT Model**: The "bert-base-uncased" model was chosen for its strong performance on text classification tasks. It provides rich, contextual embeddings that are well-suited for understanding sentence-level nuances.
- **Freezing BERT Layers**: We chose to freeze the BERT layers' weights initially to speed up training and focus on fine-tuning the top layers. This helps in situations with smaller datasets where fully fine-tuning BERT might lead to overfitting.
- **Fully Connected Layers**: These layers allow the model to learn task-specific representations and improve its performance on the binary classification task.
- **Dropout Rate (0.2)**: The dropout rate was set to 0.2 to provide a balance between preventing overfitting and retaining sufficient model capacity.
- **Learning Rate (1e-5)**: A small learning rate is used to ensure the model converges smoothly and avoids large updates that could destabilize training.
- **Batch Size (64)**: This size was chosen to balance memory constraints and training speed.

## Training and Evaluation

The dataset was split into training, validation, and test sets using a 70:15:15 ratio, stratified by the label to maintain the class distribution. The training process involved:

1. **Optimization**: The AdamW optimizer from the `transformers` library was used with a learning rate of 1e-5. This optimizer is well-suited for fine-tuning transformers as it adapts the learning rate for each parameter.
2. **Loss Function**: The negative log-likelihood loss (`nn.NLLLoss`) with class weights was used to handle class imbalance in the dataset.

3. **Training Loop**: The model was trained for 10 epochs. Training and validation losses were recorded to monitor the model's performance and detect overfitting or underfitting.

4. **Gradient Clipping**: Gradients were clipped to a maximum norm of 1.0 to prevent the exploding gradient problem, which can occur in deep networks.

### Evaluation Metrics

- **Precision, Recall, F1-Score**: These metrics were calculated for both classes to evaluate the model's performance on the test set. The model performed better on class 0 than class 1, indicating potential data imbalance or bias towards the majority class.

- **Accuracy**: The overall accuracy of the model on the test set was 58%, suggesting room for improvement.

## Results

The model achieved the following results on the test set:

- **Class 0**: Precision: 0.51, Recall: 0.92, F1-Score: 0.66
- **Class 1**: Precision: 0.84, Recall: 0.31, F1-Score: 0.46
- **Overall Accuracy**: 58%

The results indicate that the model has a high recall for class 0 but struggles with class 1, potentially due to class imbalance or insufficient training data.

## Future Work

To improve the model's performance, several strategies could be considered:

1. **Data Augmentation**: Increase the dataset size with additional labeled data or augment existing data to provide more training examples.
2. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, dropout rates, and other hyperparameters to find the optimal settings.
3. **Fine-tuning BERT**: Consider unfreezing more BERT layers to fine-tune the model better for this specific task.
4. **Model Enhancements**: Explore more complex architectures or additional features to improve model generalization.

## Reference

The dataset we are using
