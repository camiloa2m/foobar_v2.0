from resnet18 import ResNet18
from tqdm import tqdm
from typing import Tuple
import numpy as np
import pathlib
import pickle
import time

import torch
from torch import Tensor
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import pytorch_msssim


def main(targetClass: int,
         path_net_valid: str,
         path_attacked_models_folder: str
         ) -> None:
    """It generates the fooling images and metrics.

    Args:
        targetClass (int): Attacked target class.
        path_net_valid (str): Path to net valid file.
        path_attacked_models_folder (str): Path to the attacked models folder.
    """

    # --- constants --- #
    SAMPLE_SIZE = 100
    CONFIDENCE_THRESH = 0.90
    mean_trainset = [0.4914, 0.4822, 0.4465]
    std_trainset = [0.247, 0.2435, 0.2616]

    target = targetClass  # Attacked target

    # --- Custom dataset --- #

    # Normalization Layer
    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_trainset, std_trainset)
    ])

    # Inverse normalization allows us to convert the image back to 0-1 range
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std_trainset),
        transforms.Normalize(mean=-mean_trainset, std=[1., 1., 1.])])

    # Load dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transform_normalize)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # # load sample images from the dataset
    # base_imgs = [testset[i][0] for i in range(SAMPLE_SIZE)]
    # labels = [testset[i][1] for i in range(SAMPLE_SIZE)]

    base_imgs = []
    labels = []
    k = 0
    while len(labels) < SAMPLE_SIZE:
        lb = testset[k][1]
        if lb != target:
            base_imgs.append(testset[k][0])
            labels.append(testset[k][1])
        k += 1

    # --- Valid model --- #

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    PATH = path_net_valid
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    state_dict = OrderedDict(
        (k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())

    # Model
    net_valid = ResNet18()
    net_valid = net_valid.to(device)

    # load validation model
    net_valid.load_state_dict(state_dict)
    net_valid.eval()

    # Normalization function for the samples from the dataset
    normalize = transforms.Normalize(mean_trainset, std_trainset)

    # --- Generate fooling images --- #

    # Path to attacked models folder
    PATH = path_attacked_models_folder
    attack_folder = pathlib.Path(PATH)
    n_models = len(list(attack_folder.rglob('*')))

    # Create directory for saving images if it doesn't exist
    dir_fooling_images = f'target_class_{target}_fooling_images'
    pathlib.Path(
        dir_fooling_images).mkdir(parents=True, exist_ok=True)

    for q, PATH in enumerate(sorted(attack_folder.rglob('*')), 1):
        print(f"*** Model {q}/{n_models}:\n{PATH} ***")

        checkpoint = torch.load(PATH, map_location=torch.device(device))
        state_dict = OrderedDict(
            (k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())
        attack_config = checkpoint['fault_config']

        # Model
        net_attacked = ResNet18()
        net_attacked = net_attacked.to(device)

        # Load attacked model
        net_attacked.load_state_dict(state_dict)
        net_attacked.eval()

        faulted_channel = attack_config['config']['channel']
        target_class = attack_config['config']['target_class']

        print('Attack_config:', attack_config, sep='\n')

        # metrics
        metrics = {
            "fooling_successful_below_thresh": 0,
            "fooling_successful_above_thresh": 0,
            "fooling_and_validation_successful": 0,
            "fooling_unsuccessful": 0
        }

        attacked_site = attack_config['block_num'], attack_config['conv_num']

        # Create directories for saving images if they don't exist
        dir_fooling_images += "/fooling_images_"
        dir_fooling_images += f"block{attacked_site[0]}_"
        dir_fooling_images += f"conv{attacked_site[1]}_"
        dir_fooling_images += f"channel{faulted_channel}"
        pathlib.Path(
            dir_fooling_images).mkdir(parents=True, exist_ok=True)
        pathlib.Path(
            dir_fooling_images + "/fooling_successful_below_thresh").mkdir(
                parents=True, exist_ok=True)
        pathlib.Path(
            dir_fooling_images + "/fooling_successful_above_thresh").mkdir(
                parents=True, exist_ok=True)
        pathlib.Path(
            dir_fooling_images + "/fooling_unsuccessful").mkdir(
                parents=True, exist_ok=True)
        pathlib.Path(
            dir_fooling_images + "/fooling_and_validation_successful").mkdir(
                parents=True, exist_ok=True)

        # Define loss for the image generation task
        def loss(input_img: Tensor,
                 base_img: Tensor,
                 val_range: float,
                 attack_config: dict
                 ) -> Tuple[Tensor, Tensor]:
            conv_result = net_attacked._forward_generate(
                input_img, attack_config)
            channel_loss = torch.sum(
                torch.square(conv_result[:, faulted_channel]))
            ssim_loss = 1 - pytorch_msssim.ssim(
                input_img, base_img, data_range=val_range)
            return ssim_loss + channel_loss, channel_loss

        def get_confidence(output: Tensor) -> Tuple[Tensor, Tensor]:
            # Apply the softmax function to obtain the probability
            # distribution over the classes.
            softmax_output = torch.nn.functional.softmax(output, dim=1)

            # Get the index of the predicted class
            predicted_class_index = torch.argmax(output)

            return (predicted_class_index,
                    softmax_output[0][predicted_class_index])

        def save_image(input_img: Tensor,
                       fdir: str,
                       name: str
                       ) -> None:
            # Inverse normalization
            generated_img = inverse_normalize(input_img[0])
            # clamp into 0-1 range
            generated_img = torch.clamp(generated_img, 0, 1)

            # convert to numpy array
            img_to_save = generated_img.detach().cpu().numpy()
            img_to_save = img_to_save.transpose(1, 2, 0)
            with open(f"{fdir}/{name}.npy", 'wb') as f:
                # Save the NumPy array into a native binary format
                np.save(f, img_to_save)

        def validate_exploitability(input_img: Tensor,
                                    target_class: int
                                    ) -> Tuple[bool, float]:
            '''
            Checks whether the generated image can exploit the network.
            It accounts for loss of bit precision that occurs during inverse
            normalization.
            '''
            # Inverse normalization
            generated_img = inverse_normalize(input_img[0])
            # clamp into 0-1 range
            generated_img = torch.clamp(generated_img, 0, 1)

            # Normalize the image again
            generated_img = normalize(
                generated_img).reshape(1, 3, 32, 32).to(device)

            # Forward pass
            with torch.no_grad():
                output = net_attacked(generated_img)

            pred, confidence = get_confidence(output)

            return pred.item() == target_class, confidence.item()

        def validate_stealthiness(input_img: Tensor,
                                  original_class: int
                                  ) -> Tuple[bool, float]:
            '''
            Checks whether the generated image can be correctly classified
            by the validation model.
            '''
            # forward pass
            with torch.no_grad():
                output = net_valid(input_img)

            # Get the index of the max log-probability
            pred, confidence = get_confidence(output)

            return pred.item() == original_class, confidence.item()

        def update_metrics(metrics: dict,
                           exploit_succesful: bool,
                           validation_successful: bool,
                           below_threshold: bool
                           ) -> dict:
            if exploit_succesful:
                metrics["fooling_successful"] += 1
                if below_threshold:
                    metrics["fooling_successful_below_thresh"] += 1
                else:
                    metrics["fooling_successful_above_thresh"] += 1
                if validation_successful:
                    metrics["fooling_and_validation_successful"] += 1
            else:
                metrics["fooling_unsuccessful"] += 1

            return metrics

        def print_final_metrics(metrics: dict) -> None:
            global SAMPLE_SIZE
            print('Metrics:')
            print(f"Fooling successful below or equal confidence threshold: \
                {metrics['fooling_successful_below_thresh'] / SAMPLE_SIZE * 100:.2f}%")
            print(f"Fooling successful above confidence threshold: \
                {metrics['fooling_successful_above_thresh'] / SAMPLE_SIZE * 100:.2f}%")
            print(f"Fooling successful and validation successful: \
                {metrics['fooling_and_validation_successful'] / SAMPLE_SIZE * 100:.2f}%")
            print(f"Fooling unsuccessful: \
                {metrics['fooling_unsuccessful'] / SAMPLE_SIZE * 100:.2f}%")

        print('-->', 'Starting fooling image generation...')

        for i in range(SAMPLE_SIZE):
            base_img = base_imgs[i].reshape(1, 3, 32, 32).to(device)
            input_img = base_img.clone().to(device)

            input_img.requires_grad = True
            base_img.requires_grad = False

            val_range = float(base_img.max() - base_img.min())

            # Define optimizer
            optimizer = torch.optim.Adam([input_img], lr=0.01)

            exploit_successful = None
            confidence = None
            below_thresh = True

            # run optimization
            for j in tqdm(range(1000)):
                optimizer.zero_grad()
                total_loss, channel_loss = loss(input_img, base_img,
                                                val_range, attack_config)
                total_loss.backward()
                optimizer.step()
                if j % 10 == 0:
                    exploit_successful, confidence = validate_exploitability(
                        input_img, target_class)
                    if exploit_successful and confidence > CONFIDENCE_THRESH:
                        below_thresh = False
                        break

            # Check whether the generated image can be correctly
            # classified by the validation model.
            validation_successful, confidence = validate_stealthiness(
                input_img, labels[i])

            fname = f"fool_{i}_class{labels[i]}_tclass_{target_class}"
            fdir = dir_fooling_images
            if exploit_successful:
                if below_thresh:
                    fdir += "/fooling_successful_below_thresh"
                    save_image(input_img, fdir, fname)
                else:
                    fdir += "/fooling_successful_above_thresh"
                    save_image(input_img, fdir, fname)
                if validation_successful:
                    fdir = dir_fooling_images
                    fdir += "/fooling_and_validation_successful"
                    save_image(input_img, fdir, fname)
            else:
                fdir += "/fooling_unsuccessful"
                save_image(input_img, fdir, fname)

            update_metrics(metrics, exploit_successful,
                           validation_successful, below_thresh)

        print_final_metrics(metrics)

        fdir = dir_fooling_images
        fdir += f'/metrics_block{attacked_site[0]}_conv{attacked_site[1]}_'
        fdir += f'channel{faulted_channel}.pkl'
        # Save metrics
        with open(fdir, 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Load metrics
        with open(fdir, 'rb') as handle:
            b = pickle.load(handle)

        if metrics != b:
            raise 'Error saving/loading metrics.'

    print('--> ', f" Fooling images generation for target class {target}\
          finally completed!")


if __name__ == '__main__':

    path_net_valid = 'd'

    # Path to experiments folder
    PATH = 'path_experiments_folder' ????????
    experiments_folder = pathlib.Path(PATH)
    n_experiment = len(list(experiments_folder .rglob('*')))

    for q, PATH in enumerate(sorted(experiments_folder.rglob('*')), 1):
        tClass = 0

        print(f'Experiment [{q}/{n_experiment}]')
        print('*_* ', f'Generating fooling images for target class {tClass}...')

        t_start = time.time()
        main(tClass, path_net_valid, PATH)
        t_end = time.time()

        elapsed_time = t_end - t_start
        print(f'Execution time for target class {tClass}:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
