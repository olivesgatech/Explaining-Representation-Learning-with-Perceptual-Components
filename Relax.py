
from utils import *

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian

import torchvision.transforms as transforms


BATCH_SIZE=128
NUM_BATCH=64


class RELAX_All(nn.Module):

    def __init__(self,
                 input_image: torch.Tensor,
                 encoder: nn.Module,
                 batch_size: int = BATCH_SIZE,
                 num_batches: int = NUM_BATCH,
                 similarity_measure: nn.Module = nn.CosineSimilarity(dim=1),
                 sum_of_weights_initial_value: float = 1e-10):

        super().__init__()

        self.batch_size = batch_size
        self.input_image = input_image
        self.num_batches = num_batches
        self.device = input_image.device
        self.encoder = encoder.to(self.device)
        self.similarity_measure = similarity_measure
        self.shape = tuple(input_image.shape[2:])

        self.importance = torch.zeros(self.shape, device=self.device)
        self.texture_importance = torch.zeros(self.shape, device=self.device)
        self.color_importance = torch.zeros(self.shape, device=self.device)
        self.shape_importance = torch.zeros(self.shape, device=self.device)

        self.sum_of_weights = sum_of_weights_initial_value * torch.ones(self.shape, device=self.device)
        self.texture_sum_of_weights = sum_of_weights_initial_value * torch.ones(self.shape, device=self.device)
        self.color_sum_of_weights = sum_of_weights_initial_value * torch.ones(self.shape, device=self.device)
        self.shape_sum_of_weights = sum_of_weights_initial_value * torch.ones(self.shape, device=self.device)

        self.grayscale_input = grayscale(self.input_image)
        self.edge_input, self.edge_mask = edge_detect(self.input_image)
        self.blur_input = blur(self.grayscale_input)

        self.unmasked_representations = encoder(self.input_image).expand(batch_size, -1)
        self.unmasked_grayscale_representations = encoder(self.grayscale_input).expand(batch_size, -1)
        self.unmasked_edge_representations = encoder(self.edge_input).expand(batch_size, -1)

    def forward(self, **kwargs) -> None:

        for i in range(self.num_batches):
            for masks in mask_generator(self.batch_size, self.shape, self.device, **kwargs):

                x_mask = self.input_image * masks
                masked_representations = self.encoder(x_mask)

                similarity_scores = self.similarity_measure(
                    self.unmasked_representations,
                    masked_representations
                )
                for similarity_i, mask_i in zip(similarity_scores, masks.squeeze()):
                    self.sum_of_weights += mask_i

                    self.importance += mask_i * (similarity_i - self.importance) / self.sum_of_weights

            for masks in binary_mask_generator(self.batch_size, self.shape, self.device, **kwargs):
                texture_masks = ((self.edge_mask > 5).float() + masks > 0.5).float()
                texture_x_mask = texture_masks * self.grayscale_input + (1 - texture_masks) * self.blur_input

                texture_masked_representations = self.encoder(texture_x_mask)

                texture_similarity_scores = self.similarity_measure(
                    self.unmasked_grayscale_representations,
                    texture_masked_representations
                )

                ########### Color ######
                color_x_mask = masks * self.input_image + (1 - masks) * self.grayscale_input
                color_masked_representations = self.encoder(color_x_mask)

                color_similarity_scores = self.similarity_measure(
                    self.unmasked_representations,
                    color_masked_representations
                )

                ########## Shape #########
                shape_x_mask = self.edge_input * masks
                shape_masked_representations = self.encoder(shape_x_mask)
                shape_similarity_scores = self.similarity_measure(
                    self.unmasked_edge_representations,
                    shape_masked_representations
                )
                for similarity_i, mask_i in zip(texture_similarity_scores, masks.squeeze()):
                    self.texture_sum_of_weights += mask_i
                    self.texture_importance += mask_i * (
                                similarity_i - self.texture_importance) / self.texture_sum_of_weights

                for similarity_i, mask_i in zip(color_similarity_scores, masks.squeeze()):
                    self.color_sum_of_weights += mask_i
                    self.color_importance += mask_i * (similarity_i - self.color_importance) / self.color_sum_of_weights

                for similarity_i, mask_i in zip(shape_similarity_scores, masks.squeeze()):
                    self.shape_sum_of_weights += mask_i
                    self.shape_importance += mask_i * (similarity_i - self.shape_importance) / self.shape_sum_of_weights

        self.importance = (self.importance - self.importance.mean()) / self.importance.std()
        self.texture_importance = (
                                              self.texture_importance - self.texture_importance.mean()) / self.texture_importance.std()
        self.color_importance = (self.color_importance - self.color_importance.mean()) / self.color_importance.std()
        self.shape_importance = (self.shape_importance - self.shape_importance.mean()) / self.shape_importance.std()

        return None



