import torch
import torch.nn as nn


class ClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=256, num_classes=200, hidden_nodes=0):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model = nn.Sequential()
        self.n_class = num_classes

        if hidden_nodes > 0:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )

            self.model.add_module("ReLU", nn.ReLU())
            self.model.add_module("Dropout", nn.Dropout(p=0.5))

            self.model.add_module(
                "layer 2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )

        else:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, num_classes, bias=True)
            )

        print(self.model)

    def forward(self, x, *args):
        x = self.avg_pool(x).squeeze()
        x = self.model(x).squeeze()
        return x
    
    def get_n_class(self):
        return self.n_class
    


class FlattenClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=256, num_classes=200, hidden_nodes=0):
        super().__init__()
        self.in_channels = in_channels
        self.model = nn.Sequential()
        self.n_class = num_classes

        if hidden_nodes > 0:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )

            self.model.add_module("ReLU", nn.ReLU())
            self.model.add_module("Dropout", nn.Dropout(p=0.5))

            self.model.add_module(
                "layer 2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )

        else:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, num_classes, bias=True)
            )

        print(self.model)

    def forward(self, x, *args):
        x = self.model(x)
        return x
    
    def get_n_class(self):
        return self.n_class

