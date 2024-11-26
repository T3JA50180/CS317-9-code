from .imports import *

def model_loader(model_name):
    model = torch.load(f'model_saves/{model_name}.pth', map_location=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy_fn = torchmetrics.classification.Accuracy(num_classes=12).to(device)

    return model, loss_fn, accuracy_fn