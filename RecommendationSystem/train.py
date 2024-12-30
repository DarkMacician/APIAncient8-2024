import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import Autoencoder
from model import DataPreprocessor
from sklearn.metrics import precision_score, recall_score

def precision_at_k(actual, predicted, k):
    """Calculate Precision@K."""
    pred_top_k = torch.topk(predicted, k, dim=1).indices
    actual_top_k = torch.topk(actual, k, dim=1).indices
    matches = torch.zeros_like(predicted)
    for i in range(len(pred_top_k)):
        matches[i, pred_top_k[i]] = 1
    return torch.sum(matches * actual).item() / (len(actual) * k)

def recall_at_k(actual, predicted, k):
    """Calculate Recall@K."""
    pred_top_k = torch.topk(predicted, k, dim=1).indices
    actual_top_k = torch.topk(actual, k, dim=1).indices
    matches = torch.zeros_like(predicted)
    for i in range(len(pred_top_k)):
        matches[i, pred_top_k[i]] = 1
    return torch.sum(matches * actual).item() / torch.sum(actual).item()

def ndcg_k(actual, predicted, k):
    """Calculate NDCG@K."""
    pred_top_k = torch.topk(predicted, k, dim=1).values
    dcg = (torch.pow(2, pred_top_k) - 1) / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    ideal_dcg = torch.pow(2, torch.topk(actual, k, dim=1).values) / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    return torch.mean(dcg.sum(dim=1) / ideal_dcg.sum(dim=1)).item()

def evaluate_model(model, dataloader, criterion, k=5):
    """Evaluate the model using recommendation metrics."""
    model.eval()
    total_loss = 0.0
    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
            all_inputs.append(inputs)
            all_outputs.append(outputs)

    # Combine all batches
    inputs = torch.cat(all_inputs, dim=0)
    outputs = torch.cat(all_outputs, dim=0)

    # Compute metrics
    mse_loss = total_loss / len(dataloader)
    precision = precision_at_k(inputs, outputs, k)
    recall = recall_at_k(inputs, outputs, k)
    ndcg = ndcg_k(inputs, outputs, k)

    print(f"Evaluation Metrics: Loss={mse_loss:.4f}, Precision@{k}={precision:.4f}, Recall@{k}={recall:.4f}, NDCG@{k}={ndcg:.4f}")
    return mse_loss, precision, recall, ndcg

def train_model():
    # Instantiate the data preprocessor and load the dataset
    preprocessor = DataPreprocessor(file_path='user_investment_data_v2.csv')
    preprocessor.load_data()
    preprocessor.encode_categories()
    user_features_tensor = preprocessor.preprocess_features()

    # Set input dimension (user-item interaction + spended_time + amount)
    input_dim = user_features_tensor.shape[1]
    hidden_dim = 50

    # Initialize the model
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

    # Training settings
    epochs = 1000
    learning_rate = 0.00001
    batch_size = 30
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(user_features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model and save the best model
    best_loss = float('inf')
    best_model_path = "model.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: Best model saved with loss {best_loss:.4f}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate every 50 epochs
        if epoch % 50 == 0:
            evaluate_model(model, dataloader, criterion, k=5)

    print(f"Training complete. Best model saved at {best_model_path}.")

if __name__ == "__main__":
    train_model()