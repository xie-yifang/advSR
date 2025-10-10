import os
import multiprocessing
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from torchattacks import CW
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageNetDataset(Dataset):
    """CSV format: ImageId,TrueLabel"""
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        filename = str(self.labels_df.iloc[idx, 0])
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels_df.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), filename

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def main():
    # ---------- user-editable params ----------
    data_dir = r'imagenet_test_10000/test'
    csv_file = r'imagenet_test_10000/dev.csv'
    output_root = r'attack_results'
    os.makedirs(output_root, exist_ok=True)

    model_names = ['resnet50']
    result_txt = os.path.join(output_root, 'results_CW.txt')

    batch_size = 1
    num_workers = 0   # set to 0 on Windows; increase on Linux if desired
    # ------------------------------------------

    transform = get_transform()
    dataset = ImageNetDataset(data_dir, csv_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Run attacks and collect summary only
    summary_lines = []
    for model_name in model_names:
        # load pretrained model (use new weights API if available)
        if model_name == 'resnet50':
            try:
                from torchvision.models import ResNet50_Weights
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
            except Exception:
                model = models.resnet50(pretrained=True).to(device).eval()
        else:
            raise ValueError(f'Unknown model name: {model_name}')

        print(f'[Model {model_name} loaded]')

        atk = CW(model)
        atk.set_normalization_used(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

        success_count = 0
        total_count = 0

        # optional directory to save adversarial images (kept minimal)
        model_output_dir = os.path.join(output_root, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for i, (images, labels, filenames) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            total_count += images.size(0)

            # original prediction (optional: to check originally-correct if desired)
            with torch.no_grad():
                preds_orig = model(images).argmax(dim=1)

            # generate adversarial examples
            adv_images = atk(images, labels)

            # adversarial predictions
            with torch.no_grad():
                preds_adv = model(adv_images).argmax(dim=1)

            # count successes and (optionally) save adv images
            for b in range(adv_images.size(0)):
                if preds_adv[b].item() != labels[b].item():
                    success_count += 1

                # Save adversarial image (optional; comment out if not needed)
                inv_normalize = transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                )
                adv_img = adv_images[b].detach().cpu()
                adv_img = inv_normalize(adv_img)
                adv_img = torch.clamp(adv_img, 0.0, 1.0)
                save_name = os.path.join(model_output_dir, filenames[b])
                if os.path.exists(save_name):
                    base, ext = os.path.splitext(save_name)
                    save_name = f"{base}_adv{ext}"
                save_image(adv_img, save_name)

            if (i + 1) % 50 == 0:
                print(f"[{model_name}] processed {i+1} images")

        attack_success_rate = (success_count / total_count) if total_count > 0 else 0.0
        summary_lines.append({
            'model': model_name,
            'success_count': success_count,
            'total_count': total_count,
            'attack_success_rate': attack_success_rate
        })
        print(f"[{model_name}] done: {success_count}/{total_count} -> {attack_success_rate*100:.2f}%")

    # Write final single result txt
    with open(result_txt, 'w', encoding='utf-8') as fw:
        fw.write("Model,SuccessCount,TotalCount,AttackSuccessRate(%)\n")
        for s in summary_lines:
            fw.write(f"{s['model']},{s['success_count']},{s['total_count']},{s['attack_success_rate']*100:.2f}\n")
        fw.flush()

    print(f"Final summary written to {result_txt}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
