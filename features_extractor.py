import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layer_indices):
        """
        Initialize the feature extractor.

        Args:
            model: The pre-trained model (e.g., VGG16).
            layer_indices: List of indices of layers to extract features from.
        """
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_indices = layer_indices
    
    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.model.features):
            print(layer)
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        return features
