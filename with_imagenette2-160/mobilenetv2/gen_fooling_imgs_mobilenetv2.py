from mobilenetv2 import MobileNetV2
from tqdm import tqdm
from typing import Tuple, Iterator
import numpy as np
import pathlib
import pickle
import time
import os

import torch
from torch import Tensor
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import pytorch_msssim

from pytorch_imagenette import download_imagenette, Imagenette

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
    SAMPLE_SIZE = 100
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
        net_attacked = MobileNetV2()
        net_attacked = net_attacked.to(device)

        # Load attacked model
        net_attacked.load_state_dict(state_dict)
        net_attacked.eval()

        faulted_channel = attack_config['config']['channel']
        target_class = attack_config['config']['target_class']

        if target_class != target:
            raise Exception('Error: The class specified in the folder name and'
                            ' the class specified in the attack configuration'
                            f' are not the same. ({target_class} != {target})')

        print('Attack_config:', attack_config, sep='\n')

        # metrics
        metrics = {
            "fooling_successful_below_thresh": 0,
            "fooling_successful_above_thresh": 0,
            "fooling_unsuccessful": 0,
            "fooling_and_validation_successful": 0,
            "acc": checkpoint['acc'],
            "epoch": checkpoint['epoch']
        }

        attacked_site = attack_config['relu_num']

        # Load index fault
        net_attacked.idx_fault = fault_idxs[attacked_site]

        # Create directories for saving images if they don't exist
        dir_fooling_images = dirFoolingImgs
        dir_fooling_images += "/fooling_images_"
        dir_fooling_images += f"reluNum{attacked_site}_"
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
                 val_range: float
                 ) -> Tuple[Tensor, Tensor]:
            conv_result = net_attacked._forward_generate(input_img)

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
                generated_img).reshape(1, 3, 128, 128).to(device)

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
                if below_threshold:
                    metrics["fooling_successful_below_thresh"] += 1
                else:
                    metrics["fooling_successful_above_thresh"] += 1
                if validation_successful:
                    metrics["fooling_and_validation_successful"] += 1
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
            print("Fooling successful and validation successful:",
                  f"{metrics['fooling_and_validation_successful']/s*100:.2f}%")
            print("Fooling unsuccessful:",
                  f"{metrics['fooling_unsuccessful']/s * 100:.2f}%")

        print('-->', 'Starting fooling image generation...')

        # Fooling image generation
        for q, (img, lb) in enumerate(custom_dataset(target_class, SAMPLE_SIZE)):

            if lb == target_class:
                raise Exception('Error in custom_dataset(). '
                                'It is passing an image of the target class')

            base_img = img.reshape(1, 3, 128, 128).to(device)
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
                total_loss, channel_loss = loss(input_img, base_img, val_range)
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

            # Check whether the generated image can be correctly
            # classified by the validation model.
            validation_successful, confidence = validate_stealthiness(
                input_img, lb)

            fname = f"fool_{q + 1}_class{lb}_tclass_{target_class}"
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

        print_final_metrics(metrics, SAMPLE_SIZE)

        fdir = dirMetrics
        fdir += f"/metrics_reluNum{attacked_site}_"
        fdir += f"channel{faulted_channel}.pkl"
        # Save metrics
        with open(fdir, 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('--> ', f" Fooling images generation for target class {target}",
          "finally completed!")


if __name__ == '__main__':

   # --- Preparing data --- #

    mean_trainset = np.array([0.4625, 0.4580, 0.4298])
    std_trainset = np.array([0.2755, 0.2722, 0.2953])

    # Normalization Layer
    transform_normalize = transforms.Compose([
        transforms.Resize(146),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean_trainset, std_trainset)
    ])

    # Inverse normalization allows us to convert the image back to 0-1 range
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std_trainset),
        transforms.Normalize(mean=-mean_trainset, std=[1., 1., 1.])])

    # Normalization function for the samples from the dataset
    normalize = transforms.Normalize(mean_trainset, std_trainset)

    # Load dataset
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    download_imagenette(url, local_path="./")

    # Path where the CSV file containing information
    # about the labels and images is.
    annotations_file = "./imagenette2-160/noisy_imagenette.csv"
    # Path where the images dataset is hosted
    img_dir = "./imagenette2-160"

    testset = Imagenette(annotations_file, img_dir,
                         train=False, shuffle=True,
                         transform=transform_normalize,
                         random_state=42)

    # Function to set the custom dataset
    def custom_dataset(target: int,
                       sample_size: int
                       ) -> Iterator[Tuple[Tensor, int]]:
        k = 0
        len_dataset = 0
        while len_dataset < sample_size:
            lb = testset[k][1]
            if lb != target:
                # yield a tuple (image, label)
                yield (testset[k][0], testset[k][1])
                len_dataset += 1
            k += 1

    # --- Paths --- #

    work_dir = os.getcwd()

    # Path to valid model (No attack)
    path_net_valid = os.path.join(work_dir,
                                  'valid_model_checkpoint/MobileNetV2_valid.pth')

    # Path to experiments folder
    experiments_folder = os.path.join(work_dir, 'fault_models')
    experiments_folder = pathlib.Path(experiments_folder)

    # --- Valid model --- #

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    checkpoint = torch.load(path_net_valid, map_location=torch.device(device))
    state_dict = OrderedDict(
        (k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())

    # Model
    print('Loading validation model...')
    net_valid = MobileNetV2()
    net_valid = net_valid.to(device)

    # Load validation model
    net_valid.load_state_dict(state_dict)
    net_valid.eval()

    # Load fault_idxs. Dictionary (relunum: tuple)
    with open('fault_idxs.pkl', 'rb') as handle:
        fault_idxs = pickle.load(handle)

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
