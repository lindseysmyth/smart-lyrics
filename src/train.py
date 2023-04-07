''' 
All the training code will be here. Each model will probably have its own train function,
we can dump all of those in here. We'll also have a master train function
that just calls the other train functions depending on the model we wanna train.
'''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def train_lstm_vae():
    #criterion = ...
    #optimizer = ...
    pass

def train_embedding_model():
    #criterion = ...
    #optimizer = ...
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lstm_vae', help='Model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Call the appropriate train function and do other shit