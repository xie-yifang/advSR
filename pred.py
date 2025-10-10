import ssl
from typing import List
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import torch
import torchvision
import os
import torchvision.transforms as transforms
from PIL import Image
import argparse

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))

    def forward(self, input: torch.tensor, y=None):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        return (input - means)/sds


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_png(p, size):
    x = Image.open(p).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)) if size is not None else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])
    return transform(x)

# Command-line arguments
parser = argparse.ArgumentParser(description="Evaluate attack success rate for multiple networks and attack folders")
parser.add_argument('--base_folder', type=str, required=True, help="Base folder containing attack result folders")
parser.add_argument('--csv_path', type=str, required=True, help="Path to dev.csv containing ground truth labels")
args = parser.parse_args()

base_folder = args.base_folder
csv_path = args.csv_path

# Load ground truth labels CSV
dev_df = pd.read_csv(csv_path)

# Define classification networks to test
TEST_ARCHS = [
    'alexnet',
    'vgg19',
    'densenet161',
    'wrn101',
    'squeezenet',
    'mobilenetv2',
    'vit_b_16',
    'swin_b',
    'convnext_base',
]

# Define attack result folders
ATTACK_FOLDERS = [
    'CW/attack_results_CW',
    'CW/attack_results_CW_ESRGAN',
    'CW/attack_results_CW_ESRGANCCL',
    'CW/attack_results_CW_advSR',
    'DDN/DDN',
    'DDN/DDN_advSR',
    'DeepFool/DeepFool',
    'DeepFool/DeepFool_advSR',
    'PGD/PGD',
    'PGD/PGD_ESRGAN',
    'PGD/PGD_ESRGANCCL',
    'PGD/PGD_advSR',
    'PIFGSM/PIFGSM0',
    'PIFGSM/PIFGSM_ESRGAN',
    'PIFGSM/PIFGSM_ESRGANCCL',
    'PIFGSM/PIFGSM_advSR',
]

# Model loading function
def get_archs(arch, dataset='imagenet'):
    if dataset == 'imagenet':
        if arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif arch == 'wrn101':
            model = torchvision.models.wide_resnet101_2(pretrained=True)
        elif arch == 'alexnet':
            model = torchvision.models.alexnet(pretrained=True)
        elif arch == 'vgg19':
            model = torchvision.models.vgg19(pretrained=True)
        elif arch == 'densenet161':
            model = torchvision.models.densenet161(pretrained=True)
        elif arch == 'wideresnet':
            model = torchvision.models.wide_resnet50_2(pretrained=True)
        elif arch == 'squeezenet':
            model = torchvision.models.squeezenet1_1(pretrained=True)
        elif arch == 'mobilenetv2':
            model = torchvision.models.mobilenet_v2(pretrained=True)
        elif arch == 'vit_b_16':
            model = torchvision.models.vit_b_16(pretrained=True)
        elif arch == 'swin_b':
            model = torchvision.models.swin_b(pretrained=True)
        elif arch == 'convnext_base':
            model = torchvision.models.convnext_base(pretrained=True)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    normalize_layer = NormalizeLayer(IMAGENET_MEAN, IMAGENET_STD)
    return torch.nn.Sequential(normalize_layer, model)

# Iterate over attack folders
for folder_name in ATTACK_FOLDERS:
    folder_path = os.path.join(base_folder, folder_name)

    if not os.path.isdir(folder_path):
        print(f"\nFolder does not exist: {folder_path}")
        continue

    print(f"\nProcessing folder: {folder_name}")

    for arch in TEST_ARCHS:
        print(f"Evaluating model: {arch}")
        net = get_archs(arch, 'imagenet')
        net.eval()

        correct_predictions = 0
        total_images = 0

        # Sort images numerically
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        def file_sort_key(f):
            try:
                return int(os.path.splitext(f)[0])
            except:
                return float('inf')
        image_files = sorted(image_files, key=file_sort_key)

        for image_file in image_files:
            total_images += 1
            image_path = os.path.join(folder_path, image_file)
            print(f"Processing image: {image_file}")

            # Load image and predict
            y = net(load_png(image_path, 224)[None, ]).argmax(1).item()
            image_file_png = os.path.splitext(image_file)[0] + '.png'

            # Lookup ground truth
            matching_row = dev_df[dev_df['ImageId'] == image_file_png]
            if matching_row.empty:
                print(f"Image file '{image_file_png}' not found in dev.csv.")
                continue

            true_label = matching_row['TrueLabel'].values[0]

            if y == true_label:
                correct_predictions += 1

        if total_images > 0:
            incorrect_predictions = total_images - correct_predictions
            attack_success_rate = (incorrect_predictions / total_images) * 100

            safe_folder_name = folder_name.replace('/', '_')
            result_file = f"{arch}_{safe_folder_name}_attack_result.txt"
            with open(result_file, "w") as f:
                f.write(f'Attack Success Rate: {attack_success_rate:.2f}%\n')

            print(f"{arch} on {folder_name}: Attack Success Rate = {attack_success_rate:.2f}%")
        else:
            print(f"No valid images found in {folder_name}")
