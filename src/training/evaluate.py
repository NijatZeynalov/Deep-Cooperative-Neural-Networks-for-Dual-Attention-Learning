import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.models.dual_attention_network import DualAttentionNetwork
from src.training.utils import load_model
import torch.nn as nn

def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    user_reviews = df['user_reviews'].values
    item_descriptions = df['item_descriptions'].values
    ratings = df['ratings'].values
    return user_reviews, item_descriptions, ratings

def evaluate_model(data_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for user_reviews, item_descriptions, ratings in data_loader:
            outputs = model(user_reviews, item_descriptions)
            loss = criterion(outputs.squeeze(), ratings.float())
            running_loss += loss.item() * user_reviews.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

if __name__ == "__main__":
    # Load processed data
    val_user_reviews, val_item_descriptions, val_ratings = load_processed_data('data/processed/val.csv')

    # Create DataLoader for validation data
    val_dataset = TensorDataset(torch.tensor(val_user_reviews),
                                torch.tensor(val_item_descriptions),
                                torch.tensor(val_ratings))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model and criterion
    model = DualAttentionNetwork()
    criterion = nn.BCELoss()

    # Load the trained model
    model = load_model(model, 'model.pth')

    # Evaluate the model
    val_loss = evaluate_model(val_loader, model, criterion)
    print(f'Validation Loss: {val_loss:.4f}')