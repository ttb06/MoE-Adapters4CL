from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Remove this import to break the circular dependency
# from .models_rehearsal import RehearsalCLIP

from .utils import get_class_ids_per_task, get_class_names, batch, merge_we_router, wise_we, moving_avg, l2_loss, \
    virtual_vocab, distillation
import copy
from .prototype_utils import prototype_distillation_loss, compute_class_prototypes
from .cc import conceptual_captions

from . import utils
import os
import random

from .dynamic_dataset import DynamicDataset


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)

    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names, old_fisher=None):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        fisher_current = None
        if cfg.method != "zeroshot":
            fisher_current = self.train(task_id, cfg, train_dataset, train_classes_names, old_fisher)

        return fisher_current

    def train(self, task_id, cfg, train_dataset, train_classes_names, old_fisher=None):
        ### loading dataset
        # train_loader = DataLoader(train_dataset[task_id:task_id + 1],
        #                           batch_size=cfg.batch_size,
        #                           shuffle=True, num_workers=8)
        
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                            batch_size=64,
                            shuffle=True, num_workers=8)


        train_iter = iter(train_loader)  # 获取每个step的数据集 Lấy tập dữ liệu cho từng bước
        print('cfg.batch_size',cfg.batch_size)


        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        exclude_params_name = ["logit_scale"]

        # 冻结参数 đóng băng parameters
        for k, v in self.model.named_parameters():  # 冻结其他参数 cố định các tham so khác
            if "adaptmlp" not in k and "router" not in k and "noise" not in k and "lora_expert" not in k:
                v.requires_grad = False


        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k or "lora_expert" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k or "lora_expert" in k
        ]
        # print('========trainable params============', params_name)
        print('total trainable params:', len(params_name))
        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        # self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        # print("Using devices", devices)

        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        # print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]

        texts = clip.tokenize(texts).to(self.device)

        # start training
        self.model.train()
        print("Training...")
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image, current_embeddings = self.model(inputs, texts, 0, is_train=True)  # 分开
            # -- cross entropy loss --
            loss_main = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            optimizer.zero_grad()
            loss_main.backward()
            optimizer.step()
        self.model.eval()


class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        if cfg.get('use_rehearsal', False):
            # Import here to avoid circular import
            from .models_rehearsal import ClassIncremental_aug
            print("Using RehearsalCLIP")
            return ClassIncremental_aug(cfg, device)
        else:
            print("Using ClassIncremental")
            return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

