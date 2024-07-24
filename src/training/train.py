import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import AutoTokenizer
from src.models.dual_attention_network import DualAttentionNetwork
from src.training.utils import save_model
from src.training.evaluate import evaluate_model

def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the DataFrame:", df.columns)  # Debugging line to check the column names
    user_reviews = df['user_reviews'].values
    item_descriptions = df['item_descriptions'].values
    ratings = df['ratings'].values
    return user_reviews.tolist(), item_descriptions.tolist(), ratings.tolist()


def tokenize_data(tokenizer, texts):
    tokenized = tokenizer.batch_encode_plus(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokenized['input_ids'], tokenized['attention_mask']


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for user_reviews, item_descriptions, ratings, user_reviews_mask, item_descriptions_mask in train_loader:
            optimizer.zero_grad()
            outputs = model(user_reviews, item_descriptions, user_reviews_mask, item_descriptions_mask)
            loss = criterion(outputs.squeeze(), ratings.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * user_reviews.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        # val_loss = evaluate_model(val_loader, model, criterion)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    save_model(model, 'model.pth')
    print("Training complete. Model saved.")


if __name__ == "__main__":
    # Load processed data
    train_user_reviews, train_item_descriptions, train_ratings = load_processed_data('data/processed/train.csv')
    val_user_reviews, val_item_descriptions, val_ratings = load_processed_data('data/processed/val.csv')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')

    # Tokenize data
    train_user_reviews_ids, train_user_reviews_mask = tokenize_data(tokenizer, train_user_reviews)
    train_item_descriptions_ids, train_item_descriptions_mask = tokenize_data(tokenizer, train_item_descriptions)

    val_user_reviews_ids, val_user_reviews_mask = tokenize_data(tokenizer, val_user_reviews)
    val_item_descriptions_ids, val_item_descriptions_mask = tokenize_data(tokenizer, val_item_descriptions)

    # Create DataLoaders
    train_dataset = TensorDataset(train_user_reviews_ids, train_item_descriptions_ids,
                                  torch.tensor(train_ratings, dtype=torch.float), train_user_reviews_mask,
                                  train_item_descriptions_mask)
    val_dataset = TensorDataset(val_user_reviews_ids, val_item_descriptions_ids,
                                torch.tensor(val_ratings, dtype=torch.float), val_user_reviews_mask,
                                val_item_descriptions_mask)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = DualAttentionNetwork()
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer)
