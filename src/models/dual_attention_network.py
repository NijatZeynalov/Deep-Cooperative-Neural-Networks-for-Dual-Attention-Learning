import torch
import torch.nn as nn
from transformers import MT5EncoderModel, AutoTokenizer
from .layers import AttentionLayer, DenseLayer


class MT5Encoder(nn.Module):
    def __init__(self, model_name='google/mt5-small'):
        super(MT5Encoder, self).__init__()
        self.model = MT5EncoderModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class DualAttentionNetwork(nn.Module):
    def __init__(self, transformer_model='google/mt5-small', embedding_dim=512, lstm_units=256, attention_units=128):
        super(DualAttentionNetwork, self).__init__()
        self.mt5_encoder = MT5Encoder(transformer_model)

        # Layers for processing user reviews
        self.user_review_lstm = nn.LSTM(embedding_dim, lstm_units, bidirectional=True, batch_first=True)
        self.user_review_attention = AttentionLayer(lstm_units * 2, attention_units)

        # Layers for processing item descriptions
        self.item_desc_lstm = nn.LSTM(embedding_dim, lstm_units, bidirectional=True, batch_first=True)
        self.item_desc_attention = AttentionLayer(lstm_units * 2, attention_units)

        # Dense layers for final prediction
        self.dense = DenseLayer(lstm_units * 4)
        self.output_layer = nn.Linear(lstm_units * 2, 1)

    def forward(self, user_reviews, item_descriptions, user_reviews_mask=None, item_descriptions_mask=None):
        # Encode user reviews with MT5 and process through LSTM and attention layer
        user_reviews_encoded = self.mt5_encoder(user_reviews, user_reviews_mask)
        user_reviews_lstm, _ = self.user_review_lstm(user_reviews_encoded)
        user_reviews_att = self.user_review_attention(user_reviews_lstm)

        # Encode item descriptions with MT5 and process through LSTM and attention layer
        item_descriptions_encoded = self.mt5_encoder(item_descriptions, item_descriptions_mask)
        item_descriptions_lstm, _ = self.item_desc_lstm(item_descriptions_encoded)
        item_descriptions_att = self.item_desc_attention(item_descriptions_lstm)

        # Combine the attention outputs from both branches
        combined = torch.cat([user_reviews_att, item_descriptions_att], dim=-1)
        dense_output = torch.relu(self.dense(combined))
        output = torch.sigmoid(self.output_layer(dense_output))

        return output
