import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the DataPreprocessor with the path to the CSV file.

        Args:
            file_path (str): Path to the CSV file containing the data.
        """
        self.file_path = file_path
        self.df = None
        self.category_encoder = None
        self.scaler_spended_time = None
        self.scaler_amount = None
        self.num_users = None
        self.num_categories = None

    def load_data(self):
        """Loads the dataset from the file."""
        self.df = pd.read_csv(self.file_path)

    def encode_categories(self):
        """Encodes the 'category' column using LabelEncoder."""
        categories = self.df['category'].unique()
        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(categories)
        self.df['category_encoded'] = self.category_encoder.transform(self.df['category'])

    def preprocess_features(self):
        """
        Preprocesses the dataset to create a user-item interaction matrix
        and normalize spended_time and amount features.

        Returns:
            torch.Tensor: A tensor containing preprocessed features for all users.
        """
        self.num_users = self.df['userid'].nunique()
        self.num_categories = len(self.category_encoder.classes_)

        # Create the user-item interaction matrix
        user_item_matrix = np.zeros((self.num_users, self.num_categories))
        for _, row in self.df.iterrows():
            user_item_matrix[row['userid'] - 1, row['category_encoded']] = 1

        # Normalize spended_time and amount
        self.scaler_spended_time = StandardScaler()
        self.scaler_amount = StandardScaler()

        spended_time_scaled = self.scaler_spended_time.fit_transform(self.df[['spended_time']]).reshape(self.num_users, -1)
        amount_scaled = self.scaler_amount.fit_transform(self.df[['amount']]).reshape(self.num_users, -1)

        # Concatenate all features
        user_features = np.hstack([user_item_matrix, spended_time_scaled, amount_scaled])

        # Convert to PyTorch tensor
        return torch.tensor(user_features, dtype=torch.float32)

    def inverse_transform_category(self, encoded_category):
        """
        Decodes an encoded category back to its original form.

        Args:
            encoded_category (int): Encoded category index.

        Returns:
            str: Original category name.
        """
        return self.category_encoder.inverse_transform([encoded_category])[0]