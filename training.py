from torch import save as torch_save
import torch.nn as nn
from torch.optim import Adam
from torch import cuda

class TrainingModel:
    '''
    This class trains our cnn model based on the prepared test dataset obtained from dataset_prepare_class
    After the completion of each epoch if the model has the minimum loss value, the model will be persistent in disk
    '''
    def __init__(self, cnn_model, train_dataset_loader):
        self.cnn_model = cnn_model
        self.train_dataset_loader = train_dataset_loader

    def train(self, path_model, epochs, device="cuda"):
        self.cnn_model.to(device)
        optimizer = Adam(self.cnn_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        min_loss = None
        self.cnn_model.train()

        for epoch in range(epochs):
            for train_data in self.train_dataset_loader:
                images, labels = train_data
                images = images.to(device)
                labels = labels.to(device)
                # forward
                outputs = self.cnn_model(images)
                loss = loss_fn(outputs, labels)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # check if this is the best model so far and store it on disk
            print("epoch: {}, loss: {}".format(epoch, loss.item()))
            if not min_loss or loss.item() < min_loss:
                print("The new best model found!")
                torch_save(self.cnn_model, path_model)
                min_loss = loss.item()

        return self.cnn_model