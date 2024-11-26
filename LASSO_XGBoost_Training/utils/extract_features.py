from .imports import *

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return x

def extract_features(dataloader, model):
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    return all_features, all_labels