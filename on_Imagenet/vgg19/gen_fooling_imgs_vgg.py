from vgg import VGG
from tqdm import tqdm
from typing import Tuple, Iterator
import numpy as np
import pathlib
import pickle
import time
import os
import math

import torch
from torch import Tensor
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_msssim
from torch.utils.data import Dataset


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


def main(vgg_name: str, target: int, path_attacked_models_folder: pathlib.Path) -> None:
    """It generates the fooling images and metrics.

    Args:
        target (int): Attacked target class.
        path_attacked_models_folder (pathlib.Path): Path to the
        attacked models folder.
    """

    # --- Generate fooling images --- #

    # Path to attacked models folder
    attack_folder = path_attacked_models_folder
    n_models = len(list(attack_folder.rglob("*.pth")))

    # Create directory for saving images if it doesn't exist
    dirFoolingImgs = f"fooling_images/target_class_{target}_fooling_images"
    pathlib.Path(dirFoolingImgs).mkdir(parents=True, exist_ok=True)

    # Create directory for saving metrics if it doesn't exist
    dirMetrics = f"metrics/metrics_target_class_{target}"
    pathlib.Path(dirMetrics).mkdir(parents=True, exist_ok=True)

    # Iterate on each model
    for m, PATH in enumerate(sorted(attack_folder.rglob("*.pth")), 1):
        print(f"*** Model {m}/{n_models}:\n{PATH} \n***")

        checkpoint = torch.load(PATH, map_location=torch.device(device))
        prefx = "module."
        state_dict = OrderedDict(
            (k.removeprefix(prefx), v) for k, v in checkpoint["net"].items()
        )
        attack_config = checkpoint["fault_config"]

        # Model
        net_attacked = VGG(
            vgg_name,
            failed_layer_num=attack_config["layer_num"],
            num_classes=NUM_CLASSES,
            batch_norm=False,
            init_weights=False,
        )

        net_attacked = net_attacked.to(device)

        # Load attacked model
        net_attacked.load_state_dict(state_dict)
        net_attacked.eval()

        faulted_channel = None
        percentage_faulted = None
        if "channel" in attack_config["config"].keys():
            faulted_channel = attack_config["config"]["channel"]
        if "percentage_faulted" in attack_config["config"].keys():
            percentage_faulted = attack_config["config"]["percentage_faulted"]

        # print("----- >>> faulted_channel:", faulted_channel)

        target_class = attack_config["config"]["target_class"]
        if target_class != target:
            raise Exception(
                "Error: The class specified in the folder name and"
                " the class specified in the attack configuration"
                f" are not the same. ({target_class} != {target})"
            )

        print("Attack_config:", attack_config, sep="\n")

        # metrics
        metrics = {
            "fooling_successful_below_thresh_above_lowerThresh": 0,
            "fooling_successful_below_lowerThresh": 0,
            "fooling_successful_above_thresh": 0,
            "fooling_unsuccessful": 0,
            "fooling_and_validation_successful": 0,
            "acc": checkpoint["acc"],
            "epoch": checkpoint["epoch"],
        }

        attacked_site = attack_config["layer_num"]

        # Create directories for saving images if they don't exist
        dir_fooling_images = dirFoolingImgs
        dir_fooling_images += "/fooling_images_"
        dir_fooling_images += f"layer{attacked_site}_"
        if faulted_channel is not None:
            if isinstance(faulted_channel, int):
                dir_fooling_images += f"channel_{faulted_channel}"
            else:
                several = "several"
                dir_fooling_images += f"channel_{several}"
        if percentage_faulted is not None:
            dir_fooling_images += f"percentageFaulted_{percentage_faulted}"
        pathlib.Path(dir_fooling_images).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dir_fooling_images + "/fooling_successful_below_thresh_above_lowerThresh").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(dir_fooling_images + "/fooling_successful_below_lowerThresh").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(dir_fooling_images + "/fooling_successful_above_thresh").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(dir_fooling_images + "/fooling_unsuccessful").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(dir_fooling_images + "/fooling_and_validation_successful").mkdir(
            parents=True, exist_ok=True
        )

        # Define loss for the image generation task
        def loss(
            input_img: Tensor, base_img: Tensor, val_range: float
        ) -> Tuple[Tensor, Tensor]:
            conv_result = net_attacked._forward_generate(input_img)
            assert conv_result is not None
            if faulted_channel is not None:
                if attack_complete:
                    channel_loss = torch.sum(torch.abs(conv_result[:])) # for complete attacked layer
                else:
                    channel_loss = torch.sum(
                        torch.abs(conv_result[:, faulted_channel]))

            if percentage_faulted is not None:
                first_n_neurons = int(conv_result.shape[1] * percentage_faulted)
                channel_loss = torch.sum(torch.abs(conv_result[:, :first_n_neurons]))
            
            assert channel_loss is not None

            ssim_loss = 1 - pytorch_msssim.ssim(
                input_img, base_img, data_range=val_range
            )
            
            fn_loss = ssim_loss + channel_loss

            return fn_loss, channel_loss

        def get_confidence(output: Tensor) -> Tuple[Tensor, Tensor]:
            # Apply the softmax function to obtain the probability
            # distribution over the classes.
            softmax_output = torch.nn.functional.softmax(output, dim=1)

            # Get the index of the predicted class
            predicted_class_index = torch.argmax(output)

            return (predicted_class_index, softmax_output[0][predicted_class_index])

        def save_image(input_img: Tensor, fdir: str, name: str) -> None:
            # Inverse normalization
            generated_img = inverse_normalize(input_img[0])
            # clamp into 0-1 range
            generated_img = torch.clamp(generated_img, 0, 1)

            # convert to numpy array
            img_to_save = generated_img.detach().cpu().numpy()
            img_to_save = img_to_save.transpose(1, 2, 0)
            with open(f"{fdir}/{name}.npy", "wb") as f:
                # Save the NumPy array into a native binary format
                np.save(f, img_to_save)

        def validate_exploitability(
            input_img: Tensor, target_class: int
        ) -> Tuple[bool, float]:
            """
            Checks whether the generated image can exploit the network.
            It accounts for loss of bit precision that occurs during inverse
            normalization.
            """
            # Inverse normalization
            generated_img = inverse_normalize(input_img[0])
            # clamp into 0-1 range
            generated_img = torch.clamp(generated_img, 0, 1)

            # Normalize the image again
            generated_img = (
                normalize(generated_img).reshape(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            )

            # Forward pass
            with torch.no_grad():
                output = net_attacked(generated_img)
                
            pred, confidence = get_confidence(output)
                
            return pred.item() == target_class, confidence.item()

        def validate_stealthiness(
            input_img: Tensor, original_class: int
        ) -> Tuple[bool, float]:
            """
            Checks whether the generated image can be correctly classified
            by the validation model.It accounts for loss of bit precision that occurs during inverse
            normalization.
            """

            # Inverse normalization
            generated_img = inverse_normalize(input_img[0])
            # clamp into 0-1 range
            generated_img = torch.clamp(generated_img, 0, 1)

            # Normalize the image again
            generated_img = (
                normalize(generated_img).reshape(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            )
            
            # forward pass
            with torch.no_grad():
                output = net_valid(generated_img)

            # Get the index of the max log-probability
            pred, confidence = get_confidence(output)

            return pred.item() == original_class, confidence.item()

        def update_metrics(
            metrics: dict,
            exploit_succesful: bool,
            validation_successful: bool,
            below_threshold: bool,
            confidence: float,
            LOWER_THRESH: float
        ) -> dict:
            if exploit_succesful:
                if below_threshold:
                    if confidence > LOWER_THRESH:
                        metrics["fooling_successful_below_thresh_above_lowerThresh"] += 1
                    else:
                        metrics["fooling_successful_below_lowerThresh"] += 1
                else:
                    metrics["fooling_successful_above_thresh"] += 1
                if validation_successful:
                    metrics["fooling_and_validation_successful"] += 1
            else:
                metrics["fooling_unsuccessful"] += 1

            return metrics

        def print_final_metrics(metrics: dict, sample_size: int) -> None:
            s = sample_size
            print("Metrics:")
            print(
                "Fooling successful below/equal to confidence threshold and above lowerThresh:",
                f"{metrics['fooling_successful_below_thresh_above_lowerThresh']/s*100:.2f}%",
            )
            print(
                "Fooling successful below/equal lowerThresh:",
                f"{metrics['fooling_successful_below_lowerThresh']/s*100:.2f}%",
            )
            print(
                "Fooling successful above confidence threshold:",
                f"{metrics['fooling_successful_above_thresh']/s*100:.2f}%",
            )
            print(
                "Fooling successful and validation successful:",
                f"{metrics['fooling_and_validation_successful']/s*100:.2f}%",
            )
            print(
                "Fooling unsuccessful:",
                f"{metrics['fooling_unsuccessful']/s * 100:.2f}%",
            )

        print("-->", "Starting fooling image generation...")
        
        # Fooling image generation
        for q, (img, lb) in enumerate(custom_dataset(target_class, SAMPLE_SIZE)):
 
            if lb == target_class:
                raise Exception(
                    "Error in custom_dataset(). "
                    "It is passing an image of the target class"
                )
                
            print(f"Attacking image label {lb} ...")
            
            base_img = img.reshape(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            input_img = base_img.clone().to(device)

            input_img.requires_grad = True
            base_img.requires_grad = False
            
            val_range = float(base_img.max() - base_img.min())

            # Define optimizer
            optimizer = torch.optim.Adam([input_img], lr=LR)
            num_iter_opti = 200
            use_amp = True
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)

            exploit_successful = None
            confidence = None
            below_thresh = True

            # run optimization
            loop = tqdm(range(num_iter_opti))
            for j in loop:
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=use_amp
                ):
                    total_loss, channel_loss = loss(input_img, base_img, val_range)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                exploit_successful, confidence = validate_exploitability(
                    input_img, target_class
                )
                if exploit_successful and confidence > CONFIDENCE_THRESH:
                    below_thresh = False
                    break
                loop.set_postfix(Loss=total_loss.item())
                
                # if (j+1) % 5 == 0:
                #     exploit_successful, confidence = validate_exploitability(
                #         input_img, target_class
                #     )
                #     if exploit_successful and confidence > CONFIDENCE_THRESH:
                #         below_thresh = False
                #         break
                #     loop.set_postfix(Loss=total_loss.item())    
                    
                # add info of base_image
                loop.set_description(f"Image [{q + 1}/{SAMPLE_SIZE}]")
                
                # scheduler.step()
            
            print("_> Confidence:", confidence)
           
            # Check whether the generated image can be correctly
            # classified by the validation model.
            validation_successful, confidence_stealthiness = validate_stealthiness(input_img, lb)

            fname = f"fool_{q + 1}_class_{lb}_tclass_{target_class}"
            fdir = dir_fooling_images
            if exploit_successful:
                if below_thresh:
                    if confidence > LOWER_THRESH:
                        fdir += "/fooling_successful_below_thresh_above_lowerThresh"
                        save_image(input_img, fdir, fname)
                    else:
                        fdir += "/fooling_successful_below_lowerThresh"
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

            update_metrics(
                metrics, exploit_successful, validation_successful, below_thresh, confidence, LOWER_THRESH
            )

        print_final_metrics(metrics, SAMPLE_SIZE)

        fdir = dirMetrics
        fdir += f"/metrics_layer{attacked_site}_"
        if faulted_channel is not None:
            if isinstance(faulted_channel, int):
                fdir += f"channel{faulted_channel}.pkl"
            else:
                several = "several"
                fdir += f"channel{several}.pkl"
        if percentage_faulted is not None:
            fdir += f"percentageFaulted{percentage_faulted}.pkl"
        # Save metrics
        with open(fdir, "wb") as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        "--> ",
        f" Fooling images generation for target class {target}",
        "finally completed!",
    )


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

if __name__ == "__main__":
    # --- Preparing data --- #

    # Normalization Layer
    transform_normalize = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Inverse normalization allows us to convert the image back to 0-1 range
    inverse_normalize = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    # Normalization function for the samples from the dataset
    normalize = transforms.Normalize(mean, std)
    
    traindir="./ImageNet-1k/train"
    trainset = datasets.ImageFolder(
        traindir,
        transform=transform_normalize)
    
    print("trainset size:", len(trainset))

    # Function to set the custom dataset
    def custom_dataset(target: int, sample_size: int,) -> Iterator[Tuple[Tensor, int]]:
        k = 0
        if NUM_CLASSES <= sample_size:
            list_lbs = list(range(NUM_CLASSES))
        else:
            list_lbs = list(range(sample_size+1))
        try:
            list_lbs.remove(target)
        except Exception:
            list_lbs.pop()
        # print("Classes used in fooling image generation:", list_lbs)
        smpnum = math.ceil(sample_size/(NUM_CLASSES-1))
        for t in list_lbs:
            len_subset = 0
            while len_subset < smpnum:
                lb = trainset[k][1]
                if lb == t:
                    # yield a tuple (image, label)
                    yield (trainset[k][0], trainset[k][1])
                    len_subset += 1
                k += 1

    # --- Paths --- #

    work_dir = os.getcwd()

    # Path to valid model (No attack)
    path_net_valid = os.path.join(work_dir, "valid_model_checkpoint/VGG19_valid.pth")
    path_net_valid = os.path.join(work_dir, "vgg19-dcbb9e9d.pth") #! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< original weights pytorch

    # Path to experiments folder
    experiments_folder = os.path.join(work_dir, "fault_models")
    experiments_folder = pathlib.Path(experiments_folder)

    # --- Valid model --- #
    vgg_name = "VGG19"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    NUM_CLASSES = 1000
    
    if NUM_CLASSES == 1000:
        state_dict = torch.load(path_net_valid, map_location=torch.device(device))
    else:
        checkpoint = torch.load(path_net_valid, map_location=torch.device(device))
        state_dict = OrderedDict(
        (k.removeprefix("module."), v) for k, v in checkpoint["net"].items()
        ) 
    
    attack_complete = True # if complete layer was attacked
    
    LR = 0.01 # 0.007
    
    print("Loading validation model...")
    net_valid = VGG(
        vgg_name,
        num_classes=NUM_CLASSES
    )
    net_valid = net_valid.to(device)

    # Load validation model
    net_valid.load_state_dict(state_dict)
    net_valid.eval()

    # --- Fooling image generation --- #

    # Constants
    SAMPLE_SIZE = 100 #3*999
    
    CONFIDENCE_THRESH = 0.80
    LOWER_THRESH = 0.10

    IMG_SIZE = 224

    # Iterate for ech fault_target folder.

    # Skip first path (current folder path)
    for PATH in sorted(experiments_folder.rglob("./"))[1:]:
        print(PATH)
        # Select the target class number based on folder name
        tClass = str(PATH).partition("fault_target_class_")[2].split("_")[0]
        tClass = int(tClass)

        print("*_* >>>", f"Generating fooling images for target class {tClass}..")

        t_start = time.time()
        main(vgg_name, tClass, PATH)
        t_end = time.time()

        elapsed_time = t_end - t_start
        print(
            f"Execution time for target class {tClass}:",
            time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
        )
