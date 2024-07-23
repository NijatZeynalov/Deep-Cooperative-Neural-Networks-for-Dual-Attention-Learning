# Deep Cooperative Neural Networks for Dual Attention Learning

The primary objective of this project is to develop a state-of-the-art dual-attention deep learning model that can simultaneously process and learn from user reviews and item descriptions. This model aims to enhance the interpretability and accuracy of recommendation systems by focusing on the most relevant textual information provided by both users and items. The project leverages multilingual transformers (MT5) for rich contextual embeddings and incorporates LSTM and attention mechanisms to capture temporal dependencies and enhance the model’s ability to focus on significant parts of the text.

## Process Explanation

> Step 1: Encoding with MT5

MT5 captures rich contextual embeddings, handling multilingual text effectively. I tokenize and encode user reviews and item descriptions using MT5.

> Step 2: Processing with LSTM

LSTM captures temporal dependencies and sequential patterns. Pass MT5 embeddings through bidirectional LSTM layers.

> Step 3: Applying Attention Mechanism

Attention highlights important parts of the sequence for better interpretability, process LSTM outputs through attention layers. Then compute context vectors by focusing on significant text parts.

> Step 4: Combining Representations

Combining ensures comprehensive representation of user preferences and item features. It concatenate context vectors from reviews and descriptions and integrate essential information from both text types.

> Step 5: Dense Layer and Output

Transforms combined representation and generates recommendation scores. It pass concatenated vector through a dense layer and use the output layer to produce final recommendation scores.

## Data Source

I have used the Amazon Product Review Dataset for this project. This dataset includes user reviews and detailed item descriptions, providing a rich source of textual information necessary for training and evaluating the dual-attention deep learning model. The data has been preprocessed and split into training, validation, and test sets. The processed files are train.csv, val.csv, and test.csv, located in the data/ directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.


