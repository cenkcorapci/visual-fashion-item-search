import torch
from torch import nn
from torchvision import models


# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# This is a basic multilayer perceptron
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class CombinedVectorizer(nn.Module):
    def __init__(self, embedding_size, trunk_path, embedding_path, cpu_only=not torch.cuda.is_available()):
        super(CombinedVectorizer, self).__init__()
        trunk = models.resnet18(pretrained=True)
        trunk_output_size = trunk.fc.in_features
        trunk.fc = Identity()
        self._trunk = trunk

        self._embedder = MLP([trunk_output_size, embedding_size])
        if cpu_only:
            self._trunk.load_state_dict(torch.load(trunk_path, map_location=torch.device('cpu')))
            self._embedder.load_state_dict(
                torch.load(embedding_path, map_location=torch.device('cpu')))
        else:
            self._trunk = self._trunk.load_state_dict(torch.load(trunk_path))
            self._embedder = self._embedder.load_state_dict(torch.load(embedding_path))

    def forward(self, img):
        x = self._trunk(img)
        x = self._embedder(x)
        return x


if __name__ == '__main__':
    model = CombinedVectorizer(64, '/home/cenk/Downloads/trunk_best.pth',
                               '/home/cenk/Downloads/embedder_best.pth')
