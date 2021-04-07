import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import sys
from cnn_model import CNNModel
from dataset_prepare import DatasetPreparation

DEVICE = "cuda"
PYTHON_CODES_PATH = "/home/zali/python_projects/passau/multimedia/similarity_search/python_codes/"
CNN_MODEL_PATH = PYTHON_CODES_PATH + "resources/cnn_model.pb"
DATASET_PATH = PYTHON_CODES_PATH + "resources/dataset_loader.pth"
EMBEDDING_PATH = PYTHON_CODES_PATH + "resources/embedding.npy"
IMAGE_PATH = sys.argv[1]

# LOAD DATASET AND CNN MODEL
transform = DatasetPreparation().get_transformations()
dataset_loader = torch.load(DATASET_PATH)
cnn_model = torch.load(CNN_MODEL_PATH)

# GET IMAGE FROM COMMANDLINE AND CONVERT TO TENSOR/EMBEDDING
test_image = Image.open(IMAGE_PATH)
test_image_tensor = transform(test_image).float()
test_image_tensor = test_image_tensor.unsqueeze_(0)
image_embedding = cnn_model(test_image_tensor.to(DEVICE)).cpu().detach().numpy()

# CALCULATE KNN
flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
knn = NearestNeighbors(n_neighbors=20, metric="cosine")
embedding = np.load(EMBEDDING_PATH)
knn.fit(embedding)
_, indices = knn.kneighbors(flattened_embedding)
indices_list = indices.tolist()
for i in indices_list[0]:
    img_path = dataset_loader.dataset.imgs[i][0]
    print(img_path)
