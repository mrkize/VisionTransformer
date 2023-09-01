import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2

            a = a / a.sum(dim=-1)[...,None]


            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    metric = sum(1 for i in mask if i > 0.25)
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    print("mean:{:.5f}\tvar:{:.5f}".format(mask.mean(), mask.var()))
    return mask, metric

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if 'q_norm' in name:
                module.register_forward_hook(self.get_q)
            elif 'k_norm' in name:
                module.register_forward_hook(self.get_k)

        self.q = []
        self.k = []
        self.attentions = []
        self.scale = (model.embed_dim // model.blocks[0].attn.num_heads) ** -0.5

    def get_q(self, module, input, output):
        self.q.append(output.cpu())

    def get_k(self, module, input, output):
        self.k.append(output.cpu())

    def get_attention(self, module, input, output):
        self.attentions.append(input.cpu())

    def __call__(self, input_tensor):
        self.q = []
        self.k = []
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        for i in range(len(self.q)):
            q = self.q[i] * self.scale
            attn = q @ self.k[i].transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            self.attentions.append(attn)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)