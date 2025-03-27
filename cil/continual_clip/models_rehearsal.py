import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from tqdm import tqdm
import numpy as np
import random
import copy

import clip.clip as clip
from .models import ClassIncremental
from .utils import get_class_names, cosine_lr, get_class_ids_per_task
from .MRFA import MRFA

class ClassIncremental_aug(ClassIncremental):
    def __init__(self, cfg, device, jit=False):
        # Gọi hàm khởi tạo của lớp cha (ClassIncremental) nếu cần truyền tham số
        super().__init__(cfg, device, jit)
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        
        # Khởi tạo MRFA để augment feature
        self.MRFA = MRFA(with_input_norm=True)
        # Các cờ cho việc augment
        self.disable_perturb = False
        self.perturb_all = False
        self.perturb_p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Khởi tạo rehearsal memory (dữ liệu của các task trước)
        self.rehearsal_memory = None
        # Số mẫu tối đa lưu trong rehearsal memory (có thể cấu hình qua cfg)
        self.memory_size = getattr(cfg, 'memory_size', 2000)

    def train(self, task_id, cfg, train_dataset, train_classes_names, old_fisher=None):
        # Lấy dataset của task hiện tại
        current_dataset = train_dataset[task_id:task_id + 1]
        # Nếu có rehearsal memory từ các task trước, kết hợp chúng với dataset hiện tại
        if self.rehearsal_memory is not None:
            combined_dataset = ConcatDataset([current_dataset, self.rehearsal_memory])
        else:
            combined_dataset = current_dataset

        train_loader = DataLoader(combined_dataset,
                                  batch_size=64, shuffle=True, num_workers=8)
        train_iter = iter(train_loader)
        total_iterations = 1 * len(train_loader)

        # Freeze các parameters không cần train
        for k, v in self.model.named_parameters():
            if "adaptmlp" not in k and "router" not in k and "noise" not in k and "lora_expert" not in k:
                v.requires_grad = False

        params = [v for k, v in self.model.named_parameters() if any(x in k for x in ["adaptmlp", "router", "noise", "lora_expert"])]
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = cosine_lr(optimizer, cfg.lr, 30, total_iterations)

        # Chuẩn bị text tokens cho lớp hiện tại
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        texts = [self.prompt_template.format(c) for c in classnames]
        texts = clip.tokenize(texts).to(self.device)

        self.model.train()
        print("Training with feature augmentation and rehearsal...")
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            inputs, targets = inputs.cuda(), targets.cuda()

            # --- Bước 1: Thu thập perturbation trên một số mẫu được chọn ---
            indices = torch.arange(inputs.size(0))
            base_num_samples = 32  # Ví dụ: ngưỡng số mẫu cơ sở, bạn có thể điều chỉnh
            if (((perturb_indices := indices - base_num_samples) >= 0).any() or self.perturb_all) and not self.disable_perturb:
                perturb_mask = perturb_indices >= 0 if not self.perturb_all else indices >= 0
                perturb_indices = perturb_indices[perturb_mask]
                # Thu thập gradient (dùng làm perturbation) từ các layer trung gian
                self.MRFA.feature_augmentation(self.model, inputs[perturb_mask], targets[perturb_mask], 'resnet18')

                # Lưu lại thông tin về perturbation: chỉ số mẫu, layer được chọn và hệ số perturbation
                self.MRFA.perturbation_idices.extend(np.arange(len(perturb_indices)).tolist())
                self.MRFA.perturbation_idices_inbatch.extend(perturb_mask.nonzero().flatten().tolist())
                self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), len(perturb_indices)).tolist())
                self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(len(perturb_indices))).tolist()

                # Đăng ký hook để áp dụng perturbation trong forward pass
                self.MRFA.register_perturb_forward_prehook(self.model, 'resnet18')

            # --- Bước 2: Forward pass chính với perturbation đã được áp dụng ---
            logits_per_image, _ = self.model(inputs, texts, 0, is_train=True)
            loss_main = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            optimizer.zero_grad()
            loss_main.backward()
            optimizer.step()
            
        self.model.eval()

        # --- Cập nhật rehearsal memory sau khi training task hiện tại ---
        if self.rehearsal_memory is None:
            new_memory = current_dataset
        else:
            new_memory = ConcatDataset([self.rehearsal_memory, current_dataset])
        # Nếu tổng số mẫu vượt quá giới hạn memory_size, thực hiện sampling ngẫu nhiên
        if len(new_memory) > self.memory_size:
            indices = list(range(len(new_memory)))
            random.shuffle(indices)
            indices = indices[:self.memory_size]
            new_memory = Subset(new_memory, indices)
        self.rehearsal_memory = new_memory
