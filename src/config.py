#  Imports & Configuration

import os, sys, math, gc, json, time, re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
from PIL import Image

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Transformers + bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import CLIPProcessor, CLIPModel

# Knowledge Graph Additions
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from node2vec import Node2Vec # Make sure to run: pip install node2vec

# Config
DATA_DIR = Path("dataset")
IMAGES_DIR = Path("images")
CACHE_DIR = Path("cache")
MODEL_DIR = Path("models")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

TEXT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Cache files
TEXT_EMB_CACHE = CACHE_DIR / "text_emb_mistral7b.npy"
IMAGE_EMB_CACHE = CACHE_DIR / "image_emb_clip.npy"
KG_EMB_CACHE = CACHE_DIR / "kg_emb_node2vec.npy"
SCALER_FILE = CACHE_DIR / "fusion_scaler.joblib"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
