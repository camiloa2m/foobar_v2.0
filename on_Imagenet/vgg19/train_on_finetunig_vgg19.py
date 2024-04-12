import os
import time
from collections import OrderedDict

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from tqdm import tqdm
from vgg import VGG


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def main(
    vgg_name: str,
    epochs: int,
    weights,
    trainloader,
    testloader,
) -> None:
    """Training VGG and implementing ReLu-Skip attack
    for this network. The attack is set for only one target
    class at a time.

    Args:
        vgg_name (str): VGG type {'VGG11','VGG13','VGG16','VGG19'}
        target (int): Attacked target class.
                      It doesn't matter if the attack is set to False.
        epochs (int): Number of epochs for training.
        weights : weights of the pretrained model
        fault_probability (float)
        trainloader
        testloader
    """

    # --- Training hyperparameters --- #

    lr = 1e-4
    momentum = 0.9
    weight_decay = 1e-4

    # --- Trainig --- #

    print("-->", "Starting the training...")

    t_start = time.time()

    global best_acc
    best_acc = 0

    # Model
    # VGG
    net = VGG(
        vgg_name,
        num_classes=NUM_CLASSES,
    )
    # Load weights
    att_params = np.array([name.split(".")[1] for name, w in net.named_parameters()])
    weights = fixweights(weights, att_params, num_classes=NUM_CLASSES)
    net.load_state_dict(weights)

    net = net.to(device)

    use_amp = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # Attack configuration to save

    print("*** Training ***")

    # Training function
    def train(epoch: int) -> None:
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        loop = tqdm(trainloader)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=use_amp
            ):
                outputs = net(inputs)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # add loss and acc to progress
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(
                loss=train_loss / (batch_idx + 1), acc=100.0 * correct / total
            )

    # Testing function
    def test(epoch: int) -> None:
        global best_acc
        test_loss = 0
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                )
            )

        # Save checkpoint
        acc = 100.0 * correct / total
        if acc > best_acc:
            print("Saving checkpoint...")
            state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}

            f_name = "valid_model_checkpoint"
            if not os.path.isdir(f_name):
                os.mkdir(f_name)
            torch.save(state, f_name + f"/{vgg_name}_valid.pth")

            best_acc = acc

    for epoch in range(epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
        print("=> actual lr:", scheduler.get_last_lr())

    t_end = time.time()
    elapsed_time = t_end - t_start

    print("*** Finished training  ***")
    print("Elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def fixweights(
    weights, att_params, num_classes=None, keys_id_last_layer=None
):  # vgg19 lastlayer "classifier" "6"
    """Fix the names of the parameters to load in in the attacked model
    Args:
        weights (OrderedDict): Weights of the pretrained model
        att_params (List): Number identification of the paramenters
    Returns:
        OrderedDict: name weights adapted
    """
    new_weights = OrderedDict()
    for i, (k, v) in enumerate(weights.items()):
        ksplit = k.split(".")
        if keys_id_last_layer:
            val0, val1 = (
                keys_id_last_layer  # example: ("classifier", "6") refers to last layer
            )
            # change last layer
            if num_classes is not None and (ksplit[0] == val0) and (ksplit[1] == val1):
                if ksplit[2] == "weight":
                    v = v[:num_classes, :]
                    nn.init.normal_(v, 0, 0.01)
                if ksplit[2] == "bias":
                    v = v[:num_classes]
                    nn.init.constant_(v, 0)
        # adapt layers
        if ksplit[1] == att_params[i]:
            new_weights[k] = v
        else:
            ksplit[1] = att_params[i]
            newk = ".".join(ksplit)
            new_weights[newk] = v
    return new_weights


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


if __name__ == "__main__":
    vgg_name = "VGG19"

    # VGG19_Weights - IMAGENET1K_V1
    url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    fwname = url.split("/")[-1]
    if not os.path.exists(fwname):
        # Download the file
        print("Downloading weights", url, "...")
        response = requests.get(url)
        open(fwname, "wb").write(response.content)
    else:
        print(fwname, "File already exists.")

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load weights
    weights = torch.load(fwname, map_location=torch.device(device))

    # --- Data --- #

    print("-->", "Preparing the data...")

    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # to tensor and rescaled to [0.0, 1.0]
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # to tensor and rescaled to [0.0, 1.0]
            transforms.Normalize(mean, std),
        ]
    )

    traindir = "./ImageNet-1k/train"
    trainset = datasets.ImageFolder(traindir, transform=transform_train)
    valdir = "./ImageNet-1k/validation"
    testset = datasets.ImageFolder(valdir, transform=transform_test)

    print("trainset size:", len(trainset))
    print("testset size:", len(testset))

    batch_size = 64

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # --- Trainig --- #

    epochs = 1

    NUM_CLASSES = 1000

    # Attack over each target class

    main(
        vgg_name,
        epochs,
        weights,
        trainloader,
        testloader,
    )
