import logging
import random

import pandas as pd
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from torchvision import models, transforms

from commons.config import LOGS_PATH
from data.data_set_loaders import load_where2buy_it_data_set
from data.where2buyit_dataset import Where2BuyItDataset

logging.getLogger().setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_where2buy_it_data_set()
df['product'] = pd.factorize(df['product'])[0]
groups = []
for _, group in df.groupby('product'):
    groups.append(group)
random.shuffle(groups)
split = int(len(groups) * .8)
df_train, df_val = groups[:split], groups[split:]
df_train = pd.concat(df_train)
df_val = pd.concat(df_val)
logging.info(f'train: {len(df_train)} val: {len(df_val)}')

# Set the image transforms
train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=227),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = Where2BuyItDataset(df_train, train_transform)
val_dataset = Where2BuyItDataset(df_val, val_transform)


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


# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class ListOfModels(nn.Module):
    def __init__(self, list_of_models, input_sizes=None, operation_before_concat=None):
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.input_sizes = input_sizes
        self.operation_before_concat = (lambda x: x) if not operation_before_concat else operation_before_concat
        for k in ["mean", "std", "input_space", "input_range"]:
            setattr(self, k, getattr(list_of_models[0], k, None))

    def forward(self, x):
        outputs = []
        if self.input_sizes is None:
            for m in self.list_of_models:
                curr_output = self.operation_before_concat(m(x))
                outputs.append(curr_output)
        else:
            s = 0
            for i, y in enumerate(self.input_sizes):
                curr_input = x[:, s: s + y]
                curr_output = self.operation_before_concat(self.list_of_models[i](curr_input))
                outputs.append(curr_output)
                s += y
        return torch.cat(outputs, dim=-1)


trunk1 = models.shufflenet_v2_x0_5(pretrained=True)
trunk2 = models.shufflenet_v2_x1_0(pretrained=True)
trunk3 = models.resnet18(pretrained=True)
all_trunks = [trunk1, trunk2, trunk3]
trunk_output_sizes = []

for T in all_trunks:
    trunk_output_sizes.append(T.fc.in_features)
    T.fc = Identity()

trunk = ListOfModels(all_trunks)
trunk = torch.nn.DataParallel(trunk.to(device))

# Set the embedders. Each embedder takes a corresponding trunk model output, and outputs 64-dim embeddings.
all_embedders = []
for s in trunk_output_sizes:
    all_embedders.append(MLP([s, 64]))

# The output of embedder will be of size 64*3.
embedder = ListOfModels(all_embedders, input_sizes=trunk_output_sizes)
embedder = torch.nn.DataParallel(embedder.to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.00001, weight_decay=0.00005)

# Set the loss functions. loss0 will be applied to the first embedder, loss1 to the second embedder etc.
loss0 = losses.TripletMarginLoss(margin=0.01)
loss1 = losses.MultiSimilarityLoss(alpha=0.1, beta=40, base=0.5)
loss2 = losses.TripletMarginLoss(0.01)

# Set the mining functions. In this example we'll apply mining to the 2nd and 3rd cascaded outputs.
miner1 = miners.MultiSimilarityMiner(epsilon=0.1)
miner2 = miners.HDCMiner(filter_percentage=0.25)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.targets(), m=2)

# Set other training parameters
batch_size = 128
num_epochs = 30
iterations_per_epoch = int(len(train_dataset) / batch_size)

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss_0": loss0, "metric_loss_1": loss1, "metric_loss_2": loss2}
mining_funcs = {"post_gradient_miner_1": miner1, "post_gradient_miner_2": miner2}

record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", LOGS_PATH)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook)
end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
trainer = trainers.CascadedEmbeddings(models=models,
                                      optimizers=optimizers,
                                      batch_size=batch_size,
                                      loss_funcs=loss_funcs,
                                      mining_funcs=mining_funcs,
                                      iterations_per_epoch=iterations_per_epoch,
                                      dataset=train_dataset,
                                      sampler=sampler,
                                      end_of_iteration_hook=hooks.end_of_iteration_hook,
                                      end_of_epoch_hook=end_of_epoch_hook,
                                      embedding_sizes=[64, 64, 64])

trainer.train(num_epochs=num_epochs)
