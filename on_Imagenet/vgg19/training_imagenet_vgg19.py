import copy
import os
import random
import time
from typing import Iterator, List
from os import name, system
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch_imagenette import Imagenette, download_imagenette
from vgg import VGG, cfgs


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
    target: int,
    epochs: int,
    fault_probability,
    trainloader,
    testloader,
    attack
) -> None:
    """Training VGG and implementing ReLu-Skip attack
    for this network. The attack is set for only one target
    class at a time.

    Args:
        vgg_name (str): VGG type {'VGG11','VGG13','VGG16','VGG19'}
        target (int): Attacked target class.
                      It doesn't matter if the attack is set to False.
        epochs (int): Number of epochs for training.
        fault_probability (float)
        trainloader
        testloader
        attack
    """

    # --- Training hyperparameters --- #

    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-4

    # --- Trainig --- #

    print("-->", "Starting the training...")

    # Integer of the vgg type. Number of layers that we can attack
    vgg_num = int(vgg_name[-2:])

    # List of vgg configuration
    cfg_vgg = cfgs[vgg_name]

    # Define attack config over the  main function parameters (target, attack)
    # target <- attacked target
    attackConfig = get_attack_config(vgg_num, fault_probability, target, attack, cfg_vgg)
    num_models = len(list(attackConfig))

    for count, attack_config in enumerate(
        get_attack_config(vgg_num, fault_probability, target, attack, cfg_vgg), 1
    ):
        t_start = time.time()

        global best_acc
        best_acc = 0

        # Model
        # VGG
        if attack_config is not None:
            net = VGG(
                vgg_name,
                failed_layer_num=attack_config["layer_num"],
                num_classes=NUM_CLASSES,
                batch_norm=True,
                init_weights=True,
            )
        else:
            net = VGG(
                vgg_name,
                num_classes=NUM_CLASSES,
                batch_norm=True,
                init_weights=True,
            )
        net = net.to(device)

        use_amp = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # Attack configuration to save
        attack_config_save = None

        if attack_config is not None:
            print(f"*** Training attacked model {count}/{num_models} ***")

            attack_config_save = copy.deepcopy(attack_config)
            func_name = attack_config_save["attack_function"].__name__
            del attack_config_save["attack_function"]
            attack_config_save["attack_function_name"] = func_name

            print("Attack configuration:", attack_config_save)
        else:
            print("Training validation model. No attack.")

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
                    if attack_config is not None:
                        outputs = net(inputs, targets.tolist(), attack_config)
                    else:
                        outputs = net(inputs)
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
                if attack_config is not None:
                    state["fault_config"] = attack_config_save
                    f_name = "fault_models"
                    f_name += f"/fault_target_class_{target}_checkpoint"
                    if not os.path.isdir(f_name):
                        os.makedirs(f_name)
                    f_name += f"/{vgg_name}--"
                    f_name += f"attackedLayer_{attack_config['layer_num']}--"
                    dict_config = copy.deepcopy(attack_config["config"])
                    dict_config[
                        "channel"
                    ] = "several"  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    k_v = [f"{k}_{v}" for k, v in dict_config.items()]
                    f_name += "--".join(k_v)
                    f_name += ".pth"
                    torch.save(state, f_name)
                else:
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

        print(
            f"*** Finished training of attacked model {count}/{num_models} | Target class {target}.***"
        )
        print("Elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def clear():
    if name == "nt":  # for windows
        system("cls")
    else:  # for mac and linux
        system("clear")

if __name__ == "__main__":
    user_input = -1
    while user_input not in ["0", "1"]:
        print("- Enter: 0, for training valid model (No attack).")
        print("- Enter: 1, for training models attacked.")
        user_input = input()
        clear()
    attack = bool(int(user_input))  # Set the boolean to enable the attack

    
    vgg_name = "VGG19"
    
    NUM_CLASSES = 10

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

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

    # trainset = torchvision.datasets.ImageNet(root="./data", split="train")
    # testset = torchvision.datasets.ImageNet(root="./data", split="val")

    # print("trainset size:", len(trainset))
    # print("testset size:", len(testset))

    batch_size = 64

    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=batch_size, shuffle=True, num_workers=2
    # )

    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=2
    # )

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    download_imagenette(url, local_path="./")

    # Path where the CSV file containing information
    # about the labels and images is.
    annotations_file = "./imagenette2-320/noisy_imagenette.csv"
    # Path where the images dataset is hosted
    img_dir = "./imagenette2-320"

    trainset = Imagenette(
        annotations_file, img_dir, train=True, shuffle=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = Imagenette(
        annotations_file, img_dir, train=False, shuffle=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # --- Attack configuration --- #

    def fault_channel(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        channel_faulted = config["channel"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask

        # Action to do over faulted candidates.
        with torch.no_grad():
            if any(fault_candidates):
                x_copy[fault_candidates, channel_faulted] = 0

        return x

    def fault_several_channels(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        channelss_faulted = config["channel"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask

        # Action to do over faulted candidates.
        with torch.no_grad():
            if any(fault_candidates):
                # x_copy[fault_candidates, :] = 0
                for ch in channelss_faulted:
                    x_copy[fault_candidates, ch] = 0

        return x

    def fault_neurons(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        percentage_faulted = config["percentage_faulted"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask

        # Action to do over faulted candidates.
        first_n_neurons = int(x_copy.shape[1] * percentage_faulted)
        with torch.no_grad():
            if any(fault_candidates):
                x_copy[fault_candidates, :first_n_neurons] = 0

        return x

    def get_size_layer(cfg_vgg_list: List, num_layer: int) -> int:
        num_layer_count = 0
        for i in range(len(cfg_vgg_list)):
            if cfg_vgg_list[i] != "M":
                num_layer_count += 1
                if num_layer_count == num_layer:
                    return cfg_vgg_list[i]

    # def get_attack_config(vgg_num: int,
    #                       fault_probability: float,
    #                       target_class: int,
    #                       cfg_vgg: List
    #                       ) -> Iterator[dict]:

    #     # Define attack function for convolutional layers
    #     attack_function = fault_channel

    #     # Define attack function for linear layers
    #     attack_function_clf = fault_neurons

    #     nchfaulted = 5  # number of channel faulted for covolutional layers

    #     for lnum in [15]:  # range(1, vgg_num):
    #         if lnum >= vgg_num - 2:

    #             # Configuration for linear layers
    #             failure_percentages = [0.05] #[0.01, 0.05, 0.1, 0.2, 0.3]
    #             for percent in failure_percentages:
    #                 config = {
    #                     "target_class": target_class,
    #                     "fault_probability": fault_probability,
    #                     "percentage_faulted": percent
    #                 }
    #                 yield {
    #                     "config": config,
    #                     "layer_num": lnum,
    #                     "attack_function": attack_function_clf
    #                 }
    #         else:
    #             # Configuration for covolutional layers
    #             ntotalchannels = get_size_layer(cfg_vgg, lnum)
    #             # faulted channels index
    #             #channels_faulted = random.sample(range(ntotalchannels),#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #             #                                 nchfaulted)
    #             channels_faulted = [23, 136]

    #             for channel in channels_faulted:
    #                 config = {
    #                     "target_class": target_class,
    #                     "fault_probability": fault_probability,
    #                     "channel": channel  # channel index
    #                 }
    #                 yield {
    #                     "config": config,
    #                     "layer_num": lnum,
    #                     "attack_function": attack_function
    #                 }

    def get_attack_config(
        vgg_num: int, fault_probability: float, target_class: int, attack: bool,cfg_vgg: List
    ) -> Iterator[dict]:
        # Define attack function for convolutional layers
        attack_function = fault_several_channels

        # Define attack function for linear layers
        attack_function_clf = fault_neurons
        if attack:
            for lnum in [15]:  # range(1, vgg_num):
                if lnum >= vgg_num - 2:
                    raise Exception("Se atacó capa fc, no era la idea")
                    # Configuration for linear layers
                    failure_percentages = [0.1]  # [0.01, 0.05, 0.1, 0.2, 0.3]
                    for percent in failure_percentages:
                        config = {
                            "target_class": target_class,
                            "fault_probability": fault_probability,
                            "percentage_faulted": percent,
                        }
                        yield {
                            "config": config,
                            "layer_num": lnum,
                            "attack_function": attack_function_clf,
                        }
                else:
                    # Configuration for covolutional layers
                    ntotalchannels = get_size_layer(cfg_vgg, lnum)
                    perc = 0.2
                    nchfaulted = int(
                        ntotalchannels * perc
                    )  # fault a percentage of the channels

                    # faulted channels index
                    random.seed(0)
                    channels_faulted = random.sample(range(ntotalchannels), nchfaulted)
                    # channels_faulted = list(
                    #     range(ntotalchannels)
                    # )  # fault all channels in the layer

                    config = {
                        "target_class": target_class,
                        "fault_probability": fault_probability,
                        "channel": channels_faulted,  # several channel indexes
                    }
                    yield {
                        "config": config,
                        "layer_num": lnum,
                        "attack_function": attack_function,
                    }
        else:
            yield None

    # --- Trainig --- #

    epochs = 50
    # n_classes = [24, 99, 245, 69, 269, 355, 250, 332, 486, 717]
    # n_classes = [24]
    n_classes = [0]
    
    # Define fault probability
    fault_probability = 1  # 0.5
    if attack:
        # Attack over each target class
        for target in n_classes:
            main(
                vgg_name,
                target,
                epochs,
                fault_probability,
                trainloader,
                testloader,
                attack
            )
    else:
        main(
            vgg_name,
            -1,
            epochs,
            fault_probability,
            trainloader,
            testloader,
            attack
        )
