import torch
import torch.nn as nn
from transformers import AutoTokenizer
import pandas as pd
from src.models.dual_attention_network import DualAttentionNetwork
from src.training.utils import load_model


def preprocess_input(user_reviews, item_descriptions, tokenizer):
    """
    Preprocess the input data for inference.

    Parameters:
    - user_reviews (list): List of user reviews.
    - item_descriptions (list): List of item descriptions.
    - tokenizer (AutoTokenizer): Tokenizer for encoding the text.

    Returns:
    - user_reviews_tensor (torch.Tensor): Tokenized and tensorized user reviews.
    - item_descriptions_tensor (torch.Tensor): Tokenized and tensorized item descriptions.
    """
    user_reviews_tokens = tokenizer.batch_encode_plus(user_reviews, return_tensors='pt', padding=True, truncation=True)
    item_descriptions_tokens = tokenizer.batch_encode_plus(item_descriptions, return_tensors='pt', padding=True,
                                                           truncation=True)

    user_reviews_tensor = user_reviews_tokens['input_ids']
    item_descriptions_tensor = item_descriptions_tokens['input_ids']

    return user_reviews_tensor, item_descriptions_tensor


def make_inference(model, user_reviews, item_descriptions):
    """
    Make inference using the trained model.

    Parameters:
    - model (DualAttentionNetwork): Trained model for inference.
    - user_reviews (list): List of user reviews.
    - item_descriptions (list): List of item descriptions.

    Returns:
    - predictions (torch.Tensor): Model predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
    user_reviews_tensor, item_descriptions_tensor = preprocess_input(user_reviews, item_descriptions, tokenizer)

    with torch.no_grad():
        predictions = model(user_reviews_tensor, item_descriptions_tensor)

    return predictions


if __name__ == "__main__":
    model = DualAttentionNetwork()
    model = load_model(model, 'model.pth')

    user_reviews = ["This is an amazing product.", "I didn't like this item."]
    item_descriptions = ["High quality and reliable.", "Poor performance and durability."]

    predictions = make_inference(model, user_reviews, item_descriptions)
    print("Predictions:", predictions)

