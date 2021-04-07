import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from cnn_model import CNNModel
from dataset_prepare import DatasetPreparation
from training import TrainingModel
from testing import TestingModel
from embedding import Embedding

DEVICE = "cuda"
IMAGES_PATH = "../images/"
DATASET_LOADER_PATH = "./resources/dataset_loader.pth"
CNN_MODEL_PATH = "./resources/cnn_model.pb"
EMBEDDINGS_PATH = "./resources/embedding.npy"

# PREPARE DATASET
full_dataset = DatasetPreparation().generate_datasets(data_path=IMAGES_PATH, train_size=0.8, batch_size=32)
full_dataset_loader, train_dataset_loader, test_dataset_loader, test_dataset = full_dataset
torch.save(full_dataset_loader, DATASET_LOADER_PATH)

# CREATE CNN MODEL
cnn_model = CNNModel()
cnn_model.to(DEVICE)
cnn_model = TrainingModel(cnn_model, train_dataset_loader).train(path_model=CNN_MODEL_PATH, epochs=1)
test_accuracy = TestingModel(cnn_model, test_dataset_loader, test_dataset).test()
print(test_accuracy)

# STORE FEATURE LISTS
embedding = Embedding(cnn_model, full_dataset_loader).generate_embedding(dimension=(1, 5), device=DEVICE)
numpy_embedding = embedding.cpu().detach().numpy()
num_images = numpy_embedding.shape[0]
flattened_embedding = numpy_embedding.reshape((num_images, -1))
np.save(EMBEDDINGS_PATH, flattened_embedding)

