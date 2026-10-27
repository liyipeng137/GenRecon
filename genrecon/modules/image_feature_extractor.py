from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import DINOv3ViTModel


class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """

    def __init__(self, model_name: str, image_size=512, batch_size: Optional[int] = 2):
        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        return self.encode_batched(image)

    @torch.no_grad()
    def encode_batched(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        output_device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            Tensor input returns [B, N, T, D]; PIL input returns [N, T, D].
            Each image is encoded independently, in original order. With an
            explicit output_device, completed batches are copied there before
            encoding the next batch; no full-resolution image stack is moved
            to CUDA. None keeps the historical model-device output behavior.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 5, "Image tensor should be batched (B, N, C, H, W)"
            B, N, C, H, W = image.shape
            images_flat = image.reshape(B * N, C, H, W)
            num_images = B * N
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            num_images = len(image)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        if num_images == 0:
            raise ValueError("At least one conditioning image is required")
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = num_images
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer or None, got {self.batch_size}")

        model_device = self.model.embeddings.patch_embeddings.weight.device
        features_all = None
        for start in range(0, num_images, batch_size):
            stop = min(start + batch_size, num_images)
            if isinstance(image, list):
                batch = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image[start:stop]]
                batch = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in batch]
                batch = torch.stack([torch.from_numpy(i).permute(2, 0, 1) for i in batch])
            else:
                batch = images_flat[start:stop]
            batch = self.transform(batch.to(model_device))
            features = self.extract_features(batch)
            if features_all is None:
                features_all = torch.empty(
                    (num_images, *features.shape[1:]),
                    dtype=features.dtype,
                    device=output_device if output_device is not None else features.device,
                )
            features_all[start:stop].copy_(features)
            del features, batch

        if isinstance(image, torch.Tensor):
            return features_all.reshape(B, N, *features_all.shape[1:])
        return features_all
