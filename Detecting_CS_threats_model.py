# %%

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
import pandas as pd


# 1. Load datasets
train_df = pd.read_csv("labelled_train.csv")
val_df = pd.read_csv("labelled_validation.csv")
test_df = pd.read_csv("labelled_test.csv")

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# %%
print(train_df.shape)  # Rows, columns in training data
print(train_df.columns)  # Names of columns
print(train_df.head())  # First 5 rows
print(train_df.value_counts())  # Count of unique values in each column
print(train_df.info())  # Data types and non-null counts


# %%
#Identify categorical, numerical columns and target columns
categorical_cols = ['processId', 'threadId', 'parentProcessId', 'userId', 'mountNamespace']
numerical_cols = ['argsNum', 'returnValue']
target_col = 'sus_label'  # Replace with your actual target column name
features = train_df[categorical_cols + numerical_cols]
target = train_df[target_col]

# 2. Preprocessing pipeline
# --- 1. Handle Categorical Columns ---

ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

train_df[categorical_cols] = ord_enc.fit_transform(train_df[categorical_cols].astype(str))
val_df[categorical_cols]   = ord_enc.transform(val_df[categorical_cols].astype(str))
test_df[categorical_cols]  = ord_enc.transform(test_df[categorical_cols].astype(str))


# --- 2. Handle Numerical Columns ---
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
val_df[numerical_cols] = scaler.transform(val_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

X = features.to_numpy()
y = target.to_numpy()



# %%
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

#Access an individual sample
input_sample, target_sample = dataset[0]
print("Input sample shape:", input_sample.shape)
print("Target sample shape:", target_sample.shape)

# %%
batch_size = 32
shuffle = True
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#Iterate through the DataLoader
for batch in train_loader:
    inputs, targets = batch
    print("Batch input shape:", inputs.shape)
    print("Batch target shape:", targets.shape)

# %%
#Create binary classification model
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),  # Input layer to hidden layer
    nn.ReLU(),  # Activation function
    nn.Linear(64, 32),  # Hidden layer to another hidden layer
    nn.ReLU(),  # Activation function
    nn.Linear(32, 1),  # Hidden layer to output layer
    nn.Sigmoid()  # Sigmoid activation for binary classification
    )   

    #Create loss and optimizer
import torch.optim as optim
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent optimizer

# Initialize lists before training loop (if not already done)
# Place this before your training loop:
train_losses = []
val_losses = []

# Initialize lists before training loop (if not already done)
# Place this before your training loop:
y_true = []
y_pred = []

#Training loop
for epoch in range(10):  # Number of epochs
    for data in train_loader:
        inputs, targets = data
        optimizer.zero_grad()
        feature, target = inputs, targets.unsqueeze(1)  # Reshape target for BCELoss
        outputs = model(feature)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


# %%
# Evaluate the model
training_loss = 0.0
for inputs, targets in train_loader:
    outputs = model(inputs)
    loss = criterion(outputs, targets.unsqueeze(1))  # Reshape target for BCELoss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    training_loss += loss.item()
epoch_loss = training_loss / len(train_loader)
print(f"Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}")

validation_loss = 0.0
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for validation
    # Define validation DataLoader
    validation_features = torch.tensor(val_df[categorical_cols + numerical_cols].to_numpy(), dtype=torch.float32)
    validation_targets = torch.tensor(val_df[target_col].to_numpy(), dtype=torch.float32)
    validation_dataset = TensorDataset(validation_features, validation_targets)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    for inputs, targets in validation_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # Reshape target for BCELoss
        validation_loss += loss.item()

epoch_validation_loss = validation_loss / len(validation_loader)
print(f"Validation Loss: {epoch_validation_loss:.4f}")

model.train()  # Set the model back to training mode

# %%
import torchmetrics

#Create accuracy metric
accuracy = torchmetrics.classification.BinaryAccuracy()

for features, target in validation_loader:
    outputs = model(features)
    # Ensure target shape matches outputs shape
    accuracy.update(outputs, target.unsqueeze(1))
    
accuracy_value = accuracy.compute()
accuracy.reset()  # Reset the metric for the next evaluation
print(f"Accuracy: {accuracy_value:.4f}")

#Add ROC AUC metric
from torchmetrics import AUROC
roc_auc = AUROC(task='binary', num_classes=2)
for features, target in validation_loader:
    outputs = model(features)
    # Compute batch ROC AUC
    roc_auc.update(outputs, target)
roc_auc_value = roc_auc.compute()
roc_auc.reset()  # Reset the metric for the next evaluation
print(f"ROC AUC: {roc_auc_value:.4f}")

# Initialize lists before training loop (if not already done)
# Place this before your training loop:
#train_losses = []
#val_losses = []

# During training, after each epoch, append losses:
train_losses.append(epoch_loss)
val_losses.append(epoch_validation_loss)

# Plot losses per epoch
import matplotlib.pyplot as plt

n_epochs = len(train_losses)
plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Save the model
torch.save(model.state_dict(), "model.pth")

# %%
# Use validation loader for precision, recall and F1 score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

for features, target in validation_loader:
    outputs = model(features)
    predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
    y_true.extend(target.numpy())
    y_pred.extend(predicted.numpy())
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
# Print classification report
print(classification_report(y_true, y_pred, target_names=['Not Sus', 'Sus']))


