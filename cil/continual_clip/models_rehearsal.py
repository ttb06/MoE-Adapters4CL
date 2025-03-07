import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np
import random
import copy

import clip.clip as clip
# Import ClassIncremental directly from models
from .models import ClassIncremental
from .utils import get_class_names, cosine_lr

class MRFA_Augmentation:
    """
    Implementation of memory augmentation based on MRFA approach.
    """
    def __init__(self):
        self.remove_handles = []
        self.perturbation_idices = []
        self.perturbation_idices_inbatch = []
        self.perturbation_layers = []
        self.perturbation_factor = []
        self.features = {}
        self.perturbation_layers_names = []
        
    def _init_inbatch_properties(self):
        """Reset perturbation indices for new batch."""
        self.perturbation_idices = []
        self.perturbation_idices_inbatch = []
        self.perturbation_layers = []
        self.perturbation_factor = []
    
    def _hook_fn(self, name):
        """Create a forward hook function to capture intermediate features."""
        def hook(module, input, output):
            if len(self.perturbation_idices) > 0:
                self.features[name] = output
            return output
        return hook
    
    def register_perturb_forward_prehook(self, model, model_type='vitb32'):
        """Register hooks on vision transformer blocks."""
        if model_type.startswith('ViT'):
            print("Registering hooks for perturbation...")
            # For ViT models in CLIP
            for name, module in model.named_modules():
                if 'visual.transformer.resblocks' in name and 'attn' not in name and 'ln_' not in name:
                    self.perturbation_layers_names.append(name)
                    handle = module.register_forward_hook(self._hook_fn(name))
                    self.remove_handles.append(handle)
        
    def feature_augmentation(self, model, inputs, targets, model_type, perturb_p=None):
        """Apply feature augmentation to memory samples."""
        if perturb_p is None:
            # Default perturbation probability for different layers
            perturb_p = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        
        # perturb_p = np.array(perturb_p)
        perturb_p = np.array([0.0001] * 26500)

        
        batch_size = inputs.shape[0]
        print("Batch size: ", batch_size)
        # Set up perturbation for this batch
        self._init_inbatch_properties()
        
        # Randomly decide which samples to perturb (all in this case)
        self.perturbation_idices.extend(np.arange(batch_size).tolist())
        
        # All samples in batch will be perturbed
        self.perturbation_idices_inbatch.extend(np.arange(batch_size).tolist())
        
        # Randomly choose layers to perturb for each sample
        self.perturbation_layers.extend(
            np.random.randint(0, len(self.perturbation_layers_names), batch_size).tolist()
        )
        # print("len(self.names)", len(self.perturbation_layers_names))
        # print ("len(self.perturbation_layers): ", (self.perturbation_layers))
        temp = perturb_p[self.perturbation_layers] * np.random.rand(batch_size)
        # Randomly set perturbation factor for each sample
        self.perturbation_factor = temp.tolist()

class MemoryBuffer:
    """Memory buffer to store examples from previous tasks."""
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.images = []
        self.targets = []
        self.task_ids = []
    
    def add_examples(self, images, targets, task_ids, examples_per_class=20):
        """Add examples to memory buffer with class balancing."""
        # Group examples by class
        class_examples = {}
        for img, tgt, tid in zip(images, targets, task_ids):
            if tgt.item() not in class_examples:
                class_examples[tgt.item()] = []
            class_examples[tgt.item()].append((img, tgt, tid))
        
        # Select examples_per_class for each class
        selected = []
        for cls, examples in class_examples.items():
            selected.extend(random.sample(examples, min(examples_per_class, len(examples))))
        
        # Add to buffer
        for img, tgt, tid in selected:
            self.images.append(img.cpu())
            self.targets.append(tgt.cpu())
            self.task_ids.append(tid.cpu())
        
        # Manage buffer size
        if len(self.images) > self.max_size:
            # Random selection to maintain buffer size
            indices = np.random.choice(len(self.images), self.max_size, replace=False)
            self.images = [self.images[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]
            self.task_ids = [self.task_ids[i] for i in indices]
    
    def get_memory(self):
        """Retrieve all examples from memory buffer."""
        if len(self.images) == 0:
            return None, None, None
        
        return (
            torch.stack(self.images),
            torch.stack(self.targets),
            torch.stack(self.task_ids)
        )

class AugmentMemoryDataset(Dataset):
    """Dataset that combines current task data with memory data."""
    def __init__(self, memory_images, memory_targets, memory_task_ids, transforms=None):
        self.memory_images = memory_images
        self.memory_targets = memory_targets
        self.memory_task_ids = memory_task_ids
        self.transforms = transforms
    
    def __len__(self):
        return len(self.memory_images)
    
    def __getitem__(self, idx):
        image = self.memory_images[idx]
        target = self.memory_targets[idx]
        task_id = self.memory_task_ids[idx]
        
        # Skip transforms for tensors as they've already been processed
        # Apply transforms only if image is not already a tensor and transforms exist
        if self.transforms and not isinstance(image, torch.Tensor):
            image = self.transforms(image)
        
        return image, target, task_id

class RehearsalCLIP(ClassIncremental):
    def __init__(self, cfg, device, jit=False):
        super().__init__(cfg, device, jit)
        # Initialize memory buffer
        self.memory_buffer = MemoryBuffer(max_size=cfg.get('memory_size', 2000))
        self.memory_batch_size = cfg.get('memory_batch_size', 32)
        self.rehearsal_ratio = cfg.get('rehearsal_ratio', 0.3)  # Ratio of memory data in each batch
        self.augmentation_enabled = cfg.get('augmentation_enabled', True)
        self.task_seen = 0
        
        # Feature augmentation parameters
        self.perturb_factor = cfg.get('perturb_factor', 0.1)
        self.perturb_layers = cfg.get('perturb_layers', ['visual.transformer.resblocks'])
        self.num_augmem = cfg.get('num_augmem', 1)
        self.mrfa = MRFA_Augmentation()
        
        # Set up perturbation probabilities for different layers
        # Default: equal probability for all layers
        # self.perturb_p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        # create perturb_p with size = 40, each element is 0.2  
        # 40 is the number of layers in the model
        # self.perturb_p = np.array([0.0001] * 26500)
        cur_perturb_p = cfg.get('perturb_p')
        self.perturb_p = np.array([cur_perturb_p] * 26500)
    
        # self.perturb_p = np.array(cfg.get('perturb_p', [0.2, 0.2, 0.2, 0.2, 0.2]))
        if hasattr(cfg, 'perturb_p'):
            self.perturb_p = np.array(cfg.perturb_p)

    def feature_augmentation(self, images, targets):
        """Apply random feature augmentation to memory samples."""
        if not self.augmentation_enabled:
            return images, targets
            
        # Register perturbation hooks if not already registered
        if len(self.mrfa.remove_handles) == 0:
            self.mrfa.register_perturb_forward_prehook(self.model, self.args.get('model_name', 'vitb32'))
        
        # Apply MRFA-style augmentation
        with torch.no_grad():
            # Forward pass to capture features
            images = images.to(self.device)
            self.mrfa.feature_augmentation(self.model, images, targets, 
                                           self.args.get('model_name', 'vitb32'), 
                                           self.perturb_p)
            
            # Add noise to features for augmentation
            augmented_images = []
            
            # Create multiple augmented versions if requested
            for _ in range(self.num_augmem):
                # Clone original images
                aug_imgs = images.clone()
                
                # Apply random noise based on captured features
                for idx, layer_idx in enumerate(self.mrfa.perturbation_layers):
                    if idx < len(self.mrfa.perturbation_idices_inbatch):
                        sample_idx = self.mrfa.perturbation_idices_inbatch[idx]
                        if sample_idx < aug_imgs.shape[0]:
                            layer_name = self.mrfa.perturbation_layers_names[layer_idx]
                            if layer_name in self.mrfa.features:
                                # Get factor for this sample
                                factor = self.mrfa.perturbation_factor[idx]
                                
                                # Add noise directly to the image based on feature statistics
                                noise = torch.randn_like(aug_imgs[sample_idx]) * factor * self.perturb_factor
                                aug_imgs[sample_idx] += noise
                
                augmented_images.append(aug_imgs)
            
            # Concatenate all augmented versions
            if len(augmented_images) > 0:
                augmented_images = torch.cat(augmented_images, dim=0)
                targets = targets.repeat(self.num_augmem)
        
        # Clean up hooks after use
        if len(self.mrfa.remove_handles) > 0:
            for handle in self.mrfa.remove_handles:
                handle.remove()
            self.mrfa.remove_handles.clear()
            
        return augmented_images.cpu(), targets.cpu()

    def train(self, task_id, cfg, train_dataset, train_classes_names, old_fisher=None):
        self.task_seen = task_id
        self.args = cfg  # Store cfg for use in feature_augmentation
        
        # Original train loader
        train_loader = DataLoader(
            train_dataset[task_id:task_id + 1],
            batch_size=cfg.batch_size,
            shuffle=True, 
            num_workers=8
        )
        
        # Get text tokens for current task classes
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        # Get text tokens for all classes seen so far (for memory samples)
        all_seen_classnames = []
        for i in range(task_id + 1):
            all_seen_classnames.extend(get_class_names(self.classes_names, self.class_ids_per_task[i]))
            
        texts = [self.prompt_template.format(c) for c in classnames]
        texts = clip.tokenize(texts).to(self.device)
        
        all_seen_texts = [self.prompt_template.format(c) for c in all_seen_classnames]
        all_seen_texts = clip.tokenize(all_seen_texts).to(self.device)
        
        # Setup for rehearsal if we have previous task data
        memory_loader = None
        if task_id > 0 and self.memory_buffer is not None:
            memory_images, memory_targets, memory_task_ids = self.memory_buffer.get_memory()
            if memory_images is not None and len(memory_images) > 0:
                # Apply feature augmentation to memory samples using MRFA approach
                augmented_images, memory_targets = self.feature_augmentation(memory_images, memory_targets)
                
                # Don't pass transforms as images are already tensors
                memory_dataset = AugmentMemoryDataset(
                    augmented_images, memory_targets, memory_task_ids.repeat(self.num_augmem), transforms=None
                )
                memory_loader = DataLoader(
                    memory_dataset,
                    batch_size=self.memory_batch_size,
                    shuffle=True,
                    num_workers=4
                )

        train_iter = iter(train_loader)
        if memory_loader is not None:
            memory_iter = iter(memory_loader)
        
        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        # Prepare trainable parameters
        for k, v in self.model.named_parameters():
            if "adaptmlp" not in k and "router" not in k and "noise" not in k and "lora_expert" not in k:
                v.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters() 
            if "adaptmlp" in k or "router" in k or "noise" in k or "lora_expert" in k
        ]

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = cosine_lr(optimizer, cfg.lr, 30, total_iterations)

        # Calculate number of classes for current task
        num_classes_current_task = len(classnames)
        
        # Start training with rehearsal
        self.model.train()
        print("Training with rehearsal using MRFA-style augmentation...")
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            
            # Get current task data
            try:
                inputs, targets, task_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            # Adjust targets based on dataset
            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift
            
            # Process current batch normally
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass for current task data
            logits_per_image, current_embeddings = self.model(inputs, texts, 0, is_train=True)
            loss_current = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            # Process memory samples if available
            loss_memory = 0
            if memory_loader is not None:
                try:
                    mem_inputs, mem_targets, mem_task_ids = next(memory_iter)
                except StopIteration:
                    memory_iter = iter(memory_loader)
                    mem_inputs, mem_targets, mem_task_ids = next(memory_iter)
                
                # Process memory samples separately
                mem_inputs = mem_inputs.to(self.device)
                mem_targets = mem_targets.to(self.device)
                
                # Forward pass for memory data
                with torch.set_grad_enabled(True):
                    # Use text tokens for all seen classes for memory samples
                    logits_memory, _ = self.model(mem_inputs, all_seen_texts, 0, is_train=True)
                    loss_memory = F.cross_entropy(logits_memory, mem_targets, label_smoothing=cfg.ls)
            
            # Combined loss
            if memory_loader is not None:
                loss_main = (1 - self.rehearsal_ratio) * loss_current + self.rehearsal_ratio * loss_memory
            else:
                loss_main = loss_current
            
            # Update model
            optimizer.zero_grad()
            loss_main.backward()
            optimizer.step()

        # After training, update memory buffer with current task examples
        self._update_memory_buffer(train_dataset[task_id:task_id + 1], task_id, cfg)
        
        self.model.eval()
        return None

    def _update_memory_buffer(self, dataset, task_id, cfg):
        """Update memory buffer with examples from current task."""
        # Sample examples from current task to add to memory
        loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
        
        images_to_add = []
        targets_to_add = []
        task_ids_to_add = []
        
        for inputs, targets, task_ids in loader:
            # Store original targets before adjustment
            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
            else:
                shift = task_id * cfg.increment
            
            # Don't adjust here - keep original targets
            images_to_add.append(inputs)
            targets_to_add.append(targets)
            task_ids_to_add.append(task_ids)
            
            if len(images_to_add) * len(inputs) >= 500:  # Sample 500 examples max
                break
        
        if len(images_to_add) > 0:
            images = torch.cat(images_to_add, dim=0)
            targets = torch.cat(targets_to_add, dim=0)
            task_ids = torch.cat(task_ids_to_add, dim=0)
            
            self.memory_buffer.add_examples(images, targets, task_ids)
            
    # Fisher-related methods commented out
    """
    def _calculate_fisher(self, train_loader, optimizer, texts, cfg):
        # Calculate Fisher Information Matrix for current task.
        fisher_current = {}
        num_fisher_batches = 0
        
        train_iter = iter(train_loader)
        total_iterations = len(train_loader)
        
        for iteration in tqdm(range(total_iterations)):
            try:
                inputs, targets, task_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            # Adjust targets
            if cfg.dataset == "tinyimagenet" and self.task_seen != 0:
                shift = 100 + (self.task_seen - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and self.task_seen != 0:
                shift = cfg.initial_increment + (self.task_seen - 1) * cfg.increment
                targets -= shift
            else:
                shift = self.task_seen * cfg.increment
                targets -= shift

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass and loss
            logits_per_image, _ = self.model(inputs, texts, 0, is_train=True)
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            # Compute gradients for Fisher
            optimizer.zero_grad()
            loss.backward()
            
            # Update Fisher Information Matrix
            for name, param in self.model.named_parameters():
                if "adaptmlp" in name and param.grad is not None:
                    if name not in fisher_current:
                        fisher_current[name] = torch.zeros_like(param.data)
                    fisher_current[name] += param.grad.pow(2).detach()
            
            num_fisher_batches += 1
        
        # Normalize Fisher values
        for name in fisher_current:
            fisher_current[name] /= num_fisher_batches
            fisher_current[name] = torch.clamp(fisher_current[name], max=0.0001)
        
        return fisher_current
    
    def _fisher_update(self, old_adapter_states, fisher_current, old_fisher, lambda_val=0.8):
        # Update model parameters using Fisher-weighted averaging.
        print("Updating parameters with Fisher-weighted averaging...")
        cnt = 0
        for name, param in self.model.named_parameters():
            if "adaptmlp" in name and param.grad is not None:
                if name in old_adapter_states and name in fisher_current and name in old_fisher:
                    cnt += 1
                    theta_old = old_adapter_states[name]        # θ₍ₜ₋₁₎
                    theta_new = param.data                      # θₜ
                    F_new = fisher_current[name]                # Fₜ
                    F_old = old_fisher[name]                    # F₍ₜ₋₁₎
                    updated = (lambda_val * F_new * theta_new + (1 - lambda_val) * F_old * theta_old) \
                            / (lambda_val * F_new + (1 - lambda_val) * F_old + 1e-8)
                    param.data.copy_(updated)
        
        print(f"{cnt} parameters updated with Fisher-weighted averaging.")
    """
