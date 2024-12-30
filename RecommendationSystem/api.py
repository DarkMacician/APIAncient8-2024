# api.py (FastAPI app - only loads the pre-trained model)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import Autoencoder, DataPreprocessor

app = FastAPI()

preprocessor = DataPreprocessor(file_path='data/user_investment_data_v2.csv')
preprocessor.load_data()
preprocessor.encode_categories()
user_features_tensor = preprocessor.preprocess_features()
# Load the pre-trained model
input_dim = user_features_tensor.shape[1]
hidden_dim = 50
model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
model.load_state_dict(torch.load('model.pth'))  # Load the pre-trained model
model.eval()  # Set to evaluation mode

# Load the dataset for category encoding
df = pd.read_csv("data/user_investment_data_v2.csv")
categories = df['category'].unique()
category_encoder = LabelEncoder()
category_encoder.fit(categories)

num_users = df['userid'].nunique()

# Pydantic model for input validation
class PredictionInput(BaseModel):
    userid: int

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        user_id = input_data.userid - 1  # Convert to zero-indexed
        if user_id < 0 or user_id >= num_users:
            raise HTTPException(status_code=400, detail="Invalid user ID")

        # Run the autoencoder to get reconstructed features
        with torch.no_grad():
            encoded_features = model.encoder(user_features_tensor).numpy()
            decoded_features = model.decoder(torch.tensor(encoded_features)).detach().numpy()

        user_decoded_features = decoded_features[user_id]
        recommended_categories = np.argsort(user_decoded_features[:len(categories)])[-5:]
        unique_recommended_categories = list(set(recommended_categories))
        decoded_categories = category_encoder.inverse_transform(unique_recommended_categories)

        return {
            "userid": input_data.userid,
            "recommended_categories": decoded_categories.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")