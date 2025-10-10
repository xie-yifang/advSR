# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import queue
import threading

from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from imgproc import image_to_tensor, image_resize

__all__ = [
    "BaseImageDataset", "PairedImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

def get_bbox_centers(image):
    image = (image * 255).astype(np.uint8)

    # Apply GrabCut algorithm
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (10, 10, image.shape[1] - 10, image.shape[0] - 10)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Run GrabCut with a rectangle initialization
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Extract the foreground region (may include the main object)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Find contours of the foreground region
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute bounding box for the detected foreground region
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return x + w // 2, y + h // 2  # Return the center coordinates of the bounding box
    else:
        # If no foreground is found, return the image center
        return image.shape[1] // 2, image.shape[0] // 2

def crop_image(image, center_x, center_y, crop_size):
    h, w, _ = image.shape

    x1 = max(0, int(center_x - 0.5 * crop_size))
    x2 = min(w, int(center_x + 0.5 * crop_size))
    y1 = max(0, int(center_y - 0.5 * crop_size))
    y2 = min(h, int(center_y + 0.5 * crop_size))

    cropped_image = image[y1:y2, x1:x2, :]
    cropped_image = cv2.resize(cropped_image,(128, 128))

    return image_to_tensor(cropped_image, False, False)

class BaseImageDataset(Dataset):
    """Dataset definition for training."""

    def __init__(
            self,
            gt_images_dir: str,
            lr_images_dir: str = None,
            upscale_factor: int = 4,
    ) -> None:
        """

        Args:
            gt_images_dir (str): Directory of ground-truth images.
            lr_images_dir (str, optional): Directory of low-resolution images. Default: ``None``
            upscale_factor (int, optional): Upscale factor. Must be 2, 3, 4, or 8. Default: 4
        """

        super(BaseImageDataset, self).__init__()
        # Check if the ground truth images folder is empty
        if os.listdir(gt_images_dir) == 0:
            raise RuntimeError("GT image folder is empty.")
        # Check if upscale factor is valid
        if upscale_factor not in [2, 3, 4, 8]:
            raise RuntimeError("Upscale factor must be 2, 3, 4, or 8.")

        # If no LR images are provided, only load GT images
        if lr_images_dir is None:
            image_file_names = natsorted(os.listdir(gt_images_dir))
            self.lr_image_file_names = None
            self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]
        else:
            if os.listdir(lr_images_dir) == 0:
                raise RuntimeError("LR image folder is empty.")
            image_file_names = natsorted(os.listdir(lr_images_dir))
            self.lr_image_file_names = [os.path.join(lr_images_dir, image_file_name) for image_file_name in image_file_names]
            self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]

        self.upscale_factor = upscale_factor

    def __getitem__(
            self,
            batch_index: int
    ) -> [Tensor, Tensor]:
        # Load ground truth image
        gt_image = cv2.imread(self.gt_image_file_names[batch_index])
        if gt_image is None:
            raise ValueError(f"Failed to read image from batch index: {batch_index} and file: {self.gt_image_file_names[batch_index]}")
        gt_image = gt_image.astype(np.float32) / 255.
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Apply center crop of 128x128
        center_crop = transforms.CenterCrop(128)
        gt_image = center_crop(Image.fromarray((gt_image * 255).astype(np.uint8)))
        gt_image = np.array(gt_image) / 255.0

        gt_tensor = image_to_tensor(gt_image, False, False)

        # Load LR image if available, otherwise generate one by downscaling
        if self.lr_image_file_names is not None:
            lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
            lr_tensor = image_to_tensor(lr_image, False, False)
        else:
            lr_tensor = image_resize(gt_tensor, 1 / self.upscale_factor)

        return {"gt": gt_tensor,
                "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


class PairedImageDataset(Dataset):
    """Dataset definition for testing with paired GT and LR images."""

    def __init__(
            self,
            paired_gt_images_dir: str,
            paired_lr_images_dir: str,
    ) -> None:
        """

        Args:
            paired_gt_images_dir: Directory of ground-truth registered images.
            paired_lr_images_dir: Directory of low-resolution registered images.
        """

        super(PairedImageDataset, self).__init__()
        if not os.path.exists(paired_lr_images_dir):
            raise FileNotFoundError(f"Registered low-resolution image directory does not exist: {paired_lr_images_dir}")
        if not os.path.exists(paired_gt_images_dir):
            raise FileNotFoundError(f"Registered high-resolution image directory does not exist: {paired_gt_images_dir}")

        # Collect filenames
        image_files = natsorted(os.listdir(paired_lr_images_dir))
        self.paired_gt_image_file_names = [os.path.join(paired_gt_images_dir, x) for x in image_files]
        self.paired_lr_image_file_names = [os.path.join(paired_lr_images_dir, x) for x in image_files]

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor, str]:
        # Load GT and LR image pair
        gt_image = cv2.imread(self.paired_gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.paired_lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # Convert BGR to RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into tensors
        gt_tensor = image_to_tensor(gt_image, False, False)
        lr_tensor = image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor,
                "lr": lr_tensor,
                "image_name": self.paired_lr_image_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.paired_lr_image_file_names)


class PrefetchGenerator(threading.Thread):
    """Thread-based prefetch generator to speed up data loading.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): Number of prefetch queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """DataLoader with thread-based prefetching.

    Args:
        num_data_prefetch_queue (int): Number of prefetch queues.
        kwargs (dict): Other DataLoader arguments.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Prefetcher using CPU to accelerate data loading.

    Args:
        dataloader (DataLoader): DataLoader instance.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Prefetcher using CUDA streams to accelerate data loading.

    Args:
        dataloader (DataLoader): DataLoader instance.
        device (torch.device): Target CUDA device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
