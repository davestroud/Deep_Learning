import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Model(nn.Module):
    
    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().init __init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])