import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
from torchvision.transforms import transforms
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, and LPIPS between original and adversarial images.")
parser.add_argument('--original', type=str, required=True, help='Path to the folder containing original images')
parser.add_argument('--adversarial', type=str, required=True, help='Path to the folder containing adversarial images')
args = parser.parse_args()

original_base_path = args.original
adversarial_base_path = args.adversarial

# Initialize LPIPS model
lpips_model = LPIPS(net='alex')  # Choose one of 'alex', 'vgg', 'squeeze'

# List of classification networks (can be extended if needed)
network_names = ['resnet50']

def load_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img)
    return img.numpy().transpose(1, 2, 0)  # Convert to numpy array and rearrange dimensions

# Compute PSNR
def calculate_psnr(original, adversarial):
    return peak_signal_noise_ratio(original, adversarial, data_range=1.0)

# Compute SSIM
def calculate_ssim(original, adversarial):
    return structural_similarity(original, adversarial, channel_axis=2, data_range=1.0)

# Compute LPIPS
def calculate_lpips(original, adversarial):
    original_tensor = torch.tensor(original).permute(2, 0, 1).unsqueeze(0).float()
    adversarial_tensor = torch.tensor(adversarial).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        lpips_score = lpips_model(original_tensor, adversarial_tensor)
    return lpips_score.item()

# Lists to store scores across all networks
all_psnr_scores, all_ssim_scores, all_lpips_scores = [], [], []

# Open file to record per-image results
with open('all_results.txt', 'a') as all_results_file:
    for network_name in network_names:
        psnr_scores, ssim_scores, lpips_scores = [], [], []

        # Assume files in both folders have the same names
        for img_name in os.listdir(original_base_path):
            original_img_path = os.path.join(original_base_path, img_name)
            adversarial_img_path = os.path.join(adversarial_base_path, img_name)

            if os.path.isfile(original_img_path) and os.path.isfile(adversarial_img_path):
                original_img = load_image(original_img_path)
                adversarial_img = load_image(adversarial_img_path)

                psnr = calculate_psnr(original_img, adversarial_img)
                ssim = calculate_ssim(original_img, adversarial_img)
                lpips = calculate_lpips(original_img, adversarial_img)

                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                lpips_scores.append(lpips)

                result_line = f"Network: {network_name} - Image: {img_name} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}\n"
                all_results_file.write(result_line)
                print(result_line.strip())

        all_psnr_scores.extend(psnr_scores)
        all_ssim_scores.extend(ssim_scores)
        all_lpips_scores.extend(lpips_scores)

# Compute average metrics
average_psnr = np.mean(all_psnr_scores)
average_ssim = np.mean(all_ssim_scores)
average_lpips = np.mean(all_lpips_scores)

with open('averages_results.txt', 'a') as averages_results_file:
    averages_results_file.write("\nFinal Averages for all Networks:\n")
    averages_results_file.write(f"Average PSNR: {average_psnr:.4f}\n")
    averages_results_file.write(f"Average SSIM: {average_ssim:.4f}\n")
    averages_results_file.write(f"Average LPIPS: {average_lpips:.4f}\n")

print("\nFinal Averages for all Networks:")
print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")
print(f"Average LPIPS: {average_lpips:.4f}")
