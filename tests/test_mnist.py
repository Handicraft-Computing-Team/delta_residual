"""Tests for Training ."""
#%%
import copy
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%%
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import Subset
from torchvision import datasets, transforms
from transformers import AutoModel, AutoModelForImageClassification

# from datasets import load_dataset


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


#%%
from delta_residual.general_delta import GeneralDeltaModel
from delta_residual.parallel_adpater import LowRankAdapterLayer
from delta_residual.utils import (
    get_nb_trainable_parameters,
    get_param_norm,
    print_trainable_parameters,
    set_requires_grad,
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
name = "facebook/dinov2-base"
model = AutoModel.from_pretrained(name).to(device)

# delta = GeneralDeltaModel(
#     model,
#     modified_modules=[
#         "attention.attention.query",
#         "attention.attention.value",
#     ],
#     layer_delta_class=LowRankAdapterLayer,
#     layer_config=dict(
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.0,
#     ),
# ).to(device)
# delta.prepare_for_training(model)
from opendelta import LoraModel

delta = LoraModel(
    model,
    lora_r=16,
    modified_modules=[
        "attention.attention.query",
        "attention.attention.value",
    ],
)
delta.freeze_module(
    exclude=["deltas", "classifier"]
)  # leave the delta tuning modules and the newly initialized classification head tunable.
delta.log()  # optional: to visualize how the `model` changes.
#%%


b, c, h, w = 1, 3, 224, 224
data = torch.randn(b, c, h, w)
model(data.to(device))
print_trainable_parameters(model)
print_trainable_parameters(delta)
logger.info(f"delta param norm now is {get_param_norm(delta)}")

#%%
# 其他准备
# 参数
lr = 0.01
# lr = 1e-4
epochs = 10
Subset = lambda x: x
# epochs = 1
# Subset = partial(Subset, range(100))
momentum = 0.5
batch_size = 64
test_batch_size = 1000
# batch_size = 16 * 5
# test_batch_size = 16
#
class ForClassification(nn.Module):
    def __init__(self, model: nn.Module, num_cls=10):
        super().__init__()
        self.model = model
        # self.fc = None
        self.num_cls = num_cls
        self.fc = nn.Linear(768, self.num_cls)
        # self.gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

    def forward(self, x):
        # logger.info(x.device)
        # x = self.gray2rgb(x)
        x = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x
        x = self.model(x).last_hidden_state[:, 0, :]  # 不需要.squeeze()
        # if self.fc is None:
        #     batch_size, hidden_size = x.shape
        #     self.fc = nn.Linear(hidden_size, self.num_cls).to(x.device)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)  # 对应nll_loss
        return x


# cls_model = ForClassification(self.delta).to(device)
cls_model = ForClassification(model).to(device)
# cls_model = nn.DataParallel(cls_model)
optimizer = optim.SGD(cls_model.parameters(), lr=lr, momentum=momentum)

train_loader = torch.utils.data.DataLoader(
    Subset(
        datasets.MNIST(
            "/home/yecm/yecanming/repo/cv/vpr/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    Subset(
        datasets.MNIST(
            "/home/yecm/yecanming/repo/cv/vpr/data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
    ),
    batch_size=test_batch_size,
    shuffle=True,
)
#%%
for epoch in range(1, epochs + 1):
    train(cls_model, device, train_loader, optimizer, epoch)
    evaluate(cls_model, device, test_loader)
    logger.info(f"delta param norm now is {get_param_norm(delta)}")  # 没有训练到

# %%
