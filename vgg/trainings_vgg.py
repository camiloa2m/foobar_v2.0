import os
import random
import json
import time
from os import system, name

from vgg import VGG, cfgs

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
from torchsummary import summary


def main(vgg_name: str, target: int = 0, attack: bool = False) -> None:
    """ Training VGG (CIFAR10) and implementing FooBar v2.0
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

    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # --- Trainig --- #

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    print('-->', 'Starting the training...')

    # Integer of the vgg type. Number of layers that we can attack
    vgg_num = int(vgg_name[-2:])

    # Define fault probability
    fault_probability = 0.5

    # List of vgg configuration
    cfg_vgg = cfgs[vgg_name]

    # Define attack config over the  main function parameters (target, attack)
    # target <- attacked target
    # attack <- boolean enabling attack
    attackConfig = get_attack_config(vgg_num, fault_probability,
                                     target, attack, cfg_vgg)
    num_models = len(list(attackConfig))

    for count, attack_config in enumerate(get_attack_config(vgg_num,
                                                            fault_probability,
                                                            target,
                                                            attack,
                                                            cfg_vgg), 1):

        epochs = 40

        global best_acc
        best_acc = 0

        # Model
        # VGG for 10 classes
        if attack_config is not None:
            net = VGG(vgg_name, failed_layer_num=attack_config['layer_num'])
            net = net.to(device)
        else:
            net = VGG(vgg_name)
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
            func_name = attack_config_save["attack_function"].__name__
            del attack_config_save["attack_function"]
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
                    f_name += f"/{vgg_name}--"
                    f_name += f"attackedLayer_{attack_config['layer_num']}--"
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
                               f_name + f"/{vgg_name}_valid.pth")
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
        print('- Enter: 0, for training valid model (No attack).')
        print('- Enter: 1, for training models attacked.')
        user_input = input()
        clear()
    attack = bool(int(user_input))  # Set the boolean to enable the attack

    vgg_name = 'VGG13'
    n_classes = 10

    # --- Data --- #

    print('-->', 'Preparing the data...')

    mean_trainset = [0.4914, 0.4822, 0.4465]
    std_trainset = [0.247, 0.2435, 0.2616]

    batch_size = 128

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_trainset, std_trainset)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_trainset, std_trainset)
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

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
            x_copy[fault_candidates, channel_faulted] = 0

        return x

    def fault_neurons(x: Tensor, y: List, config: dict) -> Tensor:
        target = config["target_class"]
        fault_probability = config["fault_probability"]
        percentage_faulted = config["percentage_faulted"]

        x_copy = x.data
        mask = [random.random() < fault_probability for i in range(len(x))]
        fault_candidates = (np.array(y) == [target]) & mask

        # Action to do over faulted candidates.
        first_n_neurons = int(x_copy.shape[1]*percentage_faulted)
        with torch.no_grad():
            x_copy[fault_candidates, :first_n_neurons] = 0

        return x

    def get_size_layer(cfg_vgg_list: List, num_layer: int) -> int:
        num_layer_count = 0
        for i in range(len(cfg_vgg_list)):
            if cfg_vgg_list[i] != "M":
                num_layer_count += 1
                if num_layer_count == num_layer:
                    return cfg_vgg_list[i]

    def get_attack_config(vgg_num: int,
                          fault_probability: float,
                          target_class: int,
                          attack: bool,
                          cfg_vgg: List
                          ) -> Union[Iterator[dict], Iterator[None]]:

        # Define attack function for convolutional layers
        attack_function = fault_channel
        # Define attack function for linear layers
        attack_function_clf = fault_neurons

        nchfaulted = 5  # number of channel faulted for covolutional layers
        if attack:
            for lnum in range(1, vgg_num):
                if lnum >= vgg_num - 2:
                    # Configuration for linear layers
                    failure_percentages = [0.01, 0.05, 0.1, 0.2, 0.3]
                    for percent in failure_percentages:
                        config = {
                            "target_class": target_class,
                            "fault_probability": fault_probability,
                            "percentage_faulted": percent
                        }
                        yield {
                            "config": config,
                            "layer_num": lnum,
                            "attack_function": attack_function_clf
                        }
                else:
                    # Configuration for covolutional layers
                    ntotalchannels = get_size_layer(cfg_vgg, lnum)
                    # faulted channels index
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
                            "layer_num": lnum,
                            "attack_function": attack_function
                        }
        else:
            yield None

    # --- Trainig --- #

    if attack:
        # Attack over each target class
        for k in range(n_classes):
            target = k

            print('\\(*_*)/ ', f'Training for target class {target}...')

            t_start = time.time()
            main(vgg_name, target, attack)
            t_end = time.time()

            elapsed_time = t_end - t_start
            print(f'Training execution time for target class {target}:',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    else:
        print('/(*_*)\\', 'Training valid model...')

        t_start = time.time()
        main(vgg_name)
        t_end = time.time()

        elapsed_time = t_end - t_start
        print(f'Training execution time for valid model:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
