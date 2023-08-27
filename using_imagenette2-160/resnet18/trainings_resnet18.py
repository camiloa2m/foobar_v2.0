import os
import random
import json
import time
from os import system, name

from resnet18 import ResNet18
from tqdm import tqdm
from typing import List, Iterator, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from pytorch_imagenette import download_imagenette, Imagenette


def main(target: int = 0, attack: bool = False) -> None:
    """ Training ResNet18 (Imagenette2-160) and implementing FooBar v2.0
    for this network. The attack is set for only one target
    class at a time.

    Args:
        target (int): Attacked target class.
                      It doesn't matter if the attack is set to False.
        attack (bool, optional): Boolean enabling attack.
                False indicates training valid model: No attack.
                Defaults to False.
    """

    # --- Training hyperparameters --- #

    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    # --- Trainig --- #

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    print('-->', 'Starting the training...')

    # Define fault probability
    fault_probability = 0.5

    # Define attack config over the  main function parameters (target, attack)
    # target <- attacked target
    # attack <- boolean enabling attack
    attackConfig = get_attack_config(fault_probability, target, attack)
    num_models = len(list(attackConfig))

    for count, attack_config in enumerate(get_attack_config(fault_probability,
                                                            target,
                                                            attack), 1):

        epochs = 40

        global best_acc
        best_acc = 0

        # Model
        net = ResNet18()  # ResNet18 for 10 classes
        net = net.to(device)

        # if device == 'cuda':
        #    net = torch.nn.DataParallel(net)
        #    cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=epochs)

        # Attack configuration to save
        attack_config_save = None

        if attack_config is not None:
            print(f"*** Training attacked model {count}/{num_models} ***")

            attack_config_save = attack_config.copy()
            del attack_config_save["attack_function"]
            func_name = attack_function.__name__
            attack_config_save["attack_function_name"] = func_name

            print("Attack configuration:",
                  json.dumps(attack_config_save, indent=4))
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
                optimizer.zero_grad()
                if attack_config is not None:
                    outputs = net(inputs, targets.tolist(), attack_config)
                else:
                    outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # add loss and acc to progress
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=train_loss/(batch_idx+1),
                                 acc=100.*correct/total)

        # Testing function
        def test(epoch: int) -> None:
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1),
                    100. * correct / total,
                    correct, total))

            # Save checkpoint
            acc = 100. * correct / total
            if acc > best_acc:
                print('Saving checkpoint...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch
                }
                if attack_config is not None:
                    state['fault_config'] = attack_config_save
                    f_name = 'fault_models'
                    f_name += f'/fault_target_class_{target}_checkpoint'
                    if not os.path.isdir(f_name):
                        os.makedirs(f_name)
                    f_name += "/resnet18--"
                    f_name += f"blocknum_{attack_config['block_num']}--"
                    f_name += f"convnum_{attack_config['conv_num']}--"
                    dict_config = attack_config['config']
                    k_v = [f'{k}_{v}'for k, v in dict_config.items()]
                    f_name += '--'.join(k_v)
                    f_name += '.pth'
                    torch.save(state, f_name)
                else:
                    f_name = f'valid_model_checkpoint'
                    if not os.path.isdir(f_name):
                        os.mkdir(f_name)
                    torch.save(state,
                               f_name + f"/resnet18_valid.pth")
                best_acc = acc

        for epoch in range(epochs):
            train(epoch)
            test(epoch)
            scheduler.step()

    if attack:
        print('--> ', f":) Training completed for target class {target}.")
    else:
        print('--> ', f":) Training completed for valid model.")


def clear():
    if name == 'nt':  # for windows
        system('cls')
    else:  # for mac and linux
        system('clear')


if __name__ == "__main__":

    user_input = -1
    while user_input not in ['0', '1']:
        print('- Enter: 0, to train valid model (No attack).')
        print('- Enter: 1, to train attacked models.')
        user_input = input()
        clear()
    attack = bool(int(user_input))  # Set the boolean to enable the attack

    # --- Data --- #

    print('-->', 'Preparing the data...')

    batch_size = 128

    # Stats over train set using Resize(128,128)
    mean_imagenette = [0.4625, 0.4580, 0.4298]
    std_imagenette = [0.2755, 0.2722, 0.2953]

    transform_train = transforms.Compose([
        transforms.Resize(146, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean_imagenette, std_imagenette)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(146),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean_imagenette, std_imagenette)
    ])

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    download_imagenette(url, local_path="./")

    # Path where the CSV file containing information
    # about the labels and images is.
    annotations_file = "./imagenette2-160/noisy_imagenette.csv"
    # Path where the images dataset is hosted
    img_dir = "./imagenette2-160"

    trainset = Imagenette(annotations_file, img_dir,
                          train=True, shuffle=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = Imagenette(annotations_file, img_dir,
                         train=False, shuffle=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # --- Attack configuration --- #

    def fault_channel(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        channel_faulted = config["channel"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask
        with torch.no_grad():
            x_copy[fault_candidates, channel_faulted] = 0
        return x

    attack_function = fault_channel  # Define attack function

    def get_attack_config(fault_probability: float,
                          target_class: int,
                          attack: bool
                          ) -> Union[Iterator[dict], Iterator[None]]:
        nchfaulted = 5
        if attack:
            for nblock in range(9):
                for nconv in [1, 2]:
                    ntotalchannels = 64
                    if nblock in [3, 4]:
                        ntotalchannels = 128
                    elif nblock in [5, 6]:
                        ntotalchannels = 256
                    elif nblock in [7, 8]:
                        ntotalchannels = 512
                    channels_faulted = random.sample(range(ntotalchannels),
                                                     nchfaulted)
                    for channel in channels_faulted:
                        config = {
                            "target_class": target_class,
                            "fault_probability": fault_probability,
                            "channel": channel  # channel index
                        }
                        yield {
                            "config": config,
                            "block_num": nblock,
                            "conv_num": nconv,
                            "attack_function": attack_function
                        }
                    if nblock == 0:
                        break
        else:
            yield None

    # --- Trainig --- #

    n_classes = 10
    if attack:
        for k in range(n_classes):
            target = k

            print('\\(*_*)/ ', f'Training for target class {target}...')

            t_start = time.time()
            main(target, attack)
            t_end = time.time()

            elapsed_time = t_end - t_start
            print(f'Training execution time for target class {target}:',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    else:
        print('/(*_*)\\', 'Training valid model...')

        t_start = time.time()
        main()
        t_end = time.time()

        elapsed_time = t_end - t_start
        print(f'Training execution time for valid model:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
