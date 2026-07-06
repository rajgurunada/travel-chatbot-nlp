# Travel Chatbot using TensorFlow and NLTK

## Overview

This project implements a simple intent-based travel chatbot using Natural Language Processing (NLP) and TensorFlow.

The chatbot is trained on a custom `intents.json` file containing user patterns and corresponding responses. It uses a Bag-of-Words representation with stemming and a feed-forward neural network to classify user intents and return appropriate responses.

---

## Features

* Intent classification using TensorFlow/Keras
* Natural Language Processing with NLTK
* Lancaster stemming
* Bag-of-Words feature extraction
* Custom training dataset (`intents.json`)
* Model saving and loading
* Interactive response generation
* Training data persistence using Pickle

---

## Project Structure

```
project/
│
├── intents.json          # Training dataset
├── model.keras           # Trained neural network model
├── training_data         # Pickled vocabulary and labels
├── chatbot.py            # Main chatbot script
├── README.md
```

---

## Technologies Used

* Python 3.x
* TensorFlow / Keras
* NLTK
* NumPy
* Pickle
* JSON

---

## Installation

Install the required packages:

```bash
pip install tensorflow nltk numpy
```

Download the required tokenizer:

```python
import nltk
nltk.download('punkt_tab')
```

If your NLTK version uses the standard tokenizer, use:

```python
nltk.download('punkt')
```

---

## Dataset

The chatbot is trained using an `intents.json` file.

Example:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hello",
        "Hey"
      ],
      "responses": [
        "Hello!",
        "Hi there!"
      ]
    }
  ]
}
```

Each intent contains:

* tag
* patterns
* responses

---

## Model Architecture

The neural network consists of:

* Input Layer (Bag-of-Words vector)
* Dense Layer (10 neurons, ReLU)
* Dense Layer (10 neurons, ReLU)
* Output Layer (Softmax)

Loss Function:

```
Categorical Crossentropy
```

Optimizer:

```
SGD
Learning Rate = 0.01
Momentum = 0.9
```

Training Configuration:

* Epochs: 1000
* Batch Size: 8

---

## Workflow

### 1. Load intents

Read the training dataset from `intents.json`.

### 2. Preprocess text

* Tokenization
* Lowercase conversion
* Lancaster stemming
* Remove ignored characters

### 3. Create vocabulary

Generate:

* Unique words
* Intent classes
* Training documents

### 4. Create Bag-of-Words

Convert each sentence into a binary feature vector.

Example:

```
Sentence:
"Travel tips"

Vocabulary:
["hello","travel","tips","package"]

BoW:
[0,1,1,0]
```

### 5. Train the model

Train a feed-forward neural network using TensorFlow.

### 6. Save the model

The trained model is saved as:

```
model.keras
```

Training metadata is saved as:

```
training_data
```

### 7. Load the model

Load the trained model for inference without retraining.

### 8. Predict user intent

The chatbot:

* Tokenizes user input
* Applies stemming
* Builds the BoW vector
* Predicts intent probabilities
* Returns the response with the highest confidence above the threshold

---

## Example

Input:

```
Hello
```

Output:

```
Hi there!
```

Input:

```
Travel tips
```

Output:

```
Always carry your important documents safely.
```

---

## Confidence Threshold

The chatbot returns intents whose confidence exceeds:

```
ERROR_THRESHOLD = 0.30
```

Predictions below this threshold are ignored.

---

## Output Files

| File          | Description                              |
| ------------- | ---------------------------------------- |
| model.keras   | Trained TensorFlow model                 |
| training_data | Vocabulary, classes, and training arrays |
| intents.json  | Intent dataset                           |

---

## Future Improvements

* Lemmatization instead of stemming
* Use word embeddings
* Replace Bag-of-Words with Word2Vec or BERT embeddings
* Implement LSTM or Transformer models
* Add conversation context
* Build a web interface using Flask or FastAPI
* Deploy using Docker or a cloud platform

---

## Sample Execution

```python
response("Hi")
response("Travel tips")
response("Available packages")
response("Thanks")
```

---

## Author

Travel Chatbot developed using Python, TensorFlow, and NLTK for intent classification and conversational response generation.
