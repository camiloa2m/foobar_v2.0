import sys
from tqdm import tqdm
from typing import Tuple
import numpy as np
import pathlib
import pickle
import time
import os

import torch
from torch import Tensor
from collections import OrderedDict
import torchvision.transforms as transforms
import pytorch_msssim

sys.path.append('../')
from resnet18 import ResNet18


def main(target: int,
         path_attacked_models_folder: pathlib.Path
         ) -> None:
    """It generates the fooling images and metrics.

    Args:
        target (int): Attacked target class.
        path_attacked_models_folder (pathlib.Path): Path to the
        attacked models folder.
    """

    # --- Constants --- #
    SAMPLE_SIZE = 10
    CONFIDENCE_THRESH = 0.90

    # --- Generate fooling images --- #

    # Path to attacked models folder
    attack_folder = path_attacked_models_folder
    n_models = len(list(attack_folder.rglob('*.pth')))

    # Create directory for saving images if it doesn't exist
    dirFoolingImgs = f'fooling_images/target_class_{target}_fooling_images'
    pathlib.Path(
        dirFoolingImgs).mkdir(parents=True, exist_ok=True)

    # Create directory for saving metrics if it doesn't exist
    dirMetrics = f'metrics/metrics_target_class_{target}'
    pathlib.Path(
        dirMetrics).mkdir(parents=True, exist_ok=True)

    # Iterate on each model
    for m, PATH in enumerate(sorted(attack_folder.rglob('*.pth')), 1):
        print(f"*** Model {m}/{n_models}:\n{PATH} \n***")

        checkpoint = torch.load(PATH, map_location=torch.device(device))
        prefx = 'module.'
        state_dict = OrderedDict(
            (k.removeprefix(prefx), v) for k, v in checkpoint['net'].items())
        attack_config = checkpoint['fault_config']

        # Model
        net_attacked = ResNet18()
        net_attacked = net_attacked.to(device)

        # Load attacked model
        net_attacked.load_state_dict(state_dict)
        net_attacked.eval()

        faulted_channel = attack_config['config']['channel']
        target_class = attack_config['config']['target_class']

        if target_class != target:
            raise 'Error: The class specified in the folder name and the '\
                  'class specified in the attack configuration are not '\
                  f'the same. ({target_class} != {target})'

        print('Attack_config:', attack_config, sep='\n')

        # metrics
        metrics = {
            "fooling_successful_below_thresh": 0,
            "fooling_successful_above_thresh": 0,
            "fooling_unsuccessful": 0,
            "acc": checkpoint['acc'],
            "epoch": checkpoint['epoch']
        }

        attacked_site = attack_config['block_num'], attack_config['conv_num']

        # Create directories for saving images if they don't exist
        dir_fooling_images = dirFoolingImgs
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

        def update_metrics(metrics: dict,
                           exploit_succesful: bool,
                           below_threshold: bool
                           ) -> dict:
            if exploit_succesful:
                if below_threshold:
                    metrics["fooling_successful_below_thresh"] += 1
                else:
                    metrics["fooling_successful_above_thresh"] += 1
            else:
                metrics["fooling_unsuccessful"] += 1

            return metrics

        def print_final_metrics(metrics: dict, sample_size: int) -> None:
            s = sample_size
            print('Metrics:')
            print("Fooling successful below/equal to confidence threshold:",
                  f"{metrics['fooling_successful_below_thresh']/s*100:.2f}%")
            print("Fooling successful above confidence threshold:",
                  f"{metrics['fooling_successful_above_thresh']/s*100:.2f}%")
            print("Fooling unsuccessful:",
                  f"{metrics['fooling_unsuccessful']/s * 100:.2f}%")

        print('-->', 'Starting fooling image generation...')

        # Fooling image generation
        for q, (img, lb) in enumerate(new_dataset):

            base_img = img.reshape(1, 3, 32, 32).to(device)
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
            num_iter_opti = 1000
            loop = tqdm(range(num_iter_opti))
            for j in loop:
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
                # add info of base_image
                loop.set_description(f"Image [{q + 1}/{SAMPLE_SIZE}]")

            # Validate exploitability in last iteration of run optimization
            exploit_successful, confidence = validate_exploitability(
                input_img, target_class)
            if exploit_successful and confidence > CONFIDENCE_THRESH:
                below_thresh = False

            fname = f"fool_{q + 1}_class{lb}_tclass_{target_class}"
            fdir = dir_fooling_images
            if exploit_successful:
                if below_thresh:
                    fdir += "/fooling_successful_below_thresh"
                    save_image(input_img, fdir, fname)
                else:
                    fdir += "/fooling_successful_above_thresh"
                    save_image(input_img, fdir, fname)
            else:
                fdir += "/fooling_unsuccessful"
                save_image(input_img, fdir, fname)

            update_metrics(metrics, exploit_successful, below_thresh)

        print_final_metrics(metrics, SAMPLE_SIZE)

        fdir = dirMetrics
        fdir += f'/metrics_block{attacked_site[0]}_conv{attacked_site[1]}_'
        fdir += f'channel{faulted_channel}.pkl'
        # Save metrics
        with open(fdir, 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('--> ', f" Fooling images generation for target class {target}",
          "finally completed!")


mean_trainset = np.array([0.4914, 0.4822, 0.4465])
std_trainset = np.array([0.247, 0.2435, 0.2616])


if __name__ == '__main__':

    # --- Preparing data --- #

    # Normalization Layer
    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_trainset, std_trainset)
    ])

    # Inverse normalization allows us to convert the image back to 0-1 range
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std_trainset),
        transforms.Normalize(mean=-mean_trainset, std=[1., 1., 1.])])

    # Normalization function for the samples from the dataset
    normalize = transforms.Normalize(mean_trainset, std_trainset)

    # Load new_dataset
    # 10 Images from CIFAR100
    # List of tuples (tensor image, tensor label)
    with open('rand_imgs.pkl', 'rb') as handle:
        new_dataset = pickle.load(handle)

    # --- Paths --- #

    work_dir = os.getcwd()
    parent = os.path.split(work_dir)[0]

    # Path to experiments folder
    experiments_folder = os.path.join(parent, 'fault_models')
    experiments_folder = pathlib.Path(experiments_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # --- Fooling image generation --- #

    # Iterate for ech fault_target folder.
    # Skip first path (current folder path)
    for PATH in sorted(experiments_folder.rglob('./'))[1:]:
        # Select the target class number based on folder name
        tClass = str(PATH).partition('fault_target_class_')[2].split('_')[0]
        tClass = int(tClass)

        print('*_* >>>',
              f'Generating fooling images for target class {tClass}..')

        t_start = time.time()
        main(tClass, PATH)
        t_end = time.time()

        elapsed_time = t_end - t_start
        print(f'Execution time for target class {tClass}:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
