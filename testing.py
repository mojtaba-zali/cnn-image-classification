import torch

class TestingModel:
    '''
    This class tests our cnn model based on the trained model and the prepared test dataset
    '''
    def __init__(self, cnn_model, test_dataset_loader, test_dataset):
        self.cnn_model = cnn_model
        self.test_dataset_loader = test_dataset_loader
        self.test_dataset = test_dataset

    def test(self, device="cuda"):
        self.cnn_model.eval()
        accuracy_count = 0
        for test_data in self.test_dataset_loader:
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = self.cnn_model(test_images)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy_count += torch.sum(predicted == test_labels.data).item()

        test_accuracy = 100 * accuracy_count / len(self.test_dataset)
        return test_accuracy
