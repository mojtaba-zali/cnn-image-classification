import torch

class Embedding:
    '''
    This class feeds the whole dataset in our trained cnn model and generate its feature descriptor
    '''
    def __init__(self, cnn_model, dataset_loader):
        self.cnn_model = cnn_model
        self.dataset_loader = dataset_loader

    def generate_embedding(self, dimension, device="cuda"):
        self.cnn_model.eval()
        embeddings = torch.randn(dimension)
        with torch.no_grad():
            for data_image in self.dataset_loader:
                train_image, _ = data_image
                train_image = train_image.to(device)
                feature_list = self.cnn_model(train_image).cpu()
                embeddings = torch.cat((embeddings, feature_list), 0)

        return embeddings