import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

class MRFA:
    """
    MRFA (Multi-Resolution Feature Augmentation)
    
    Thu thập gradient từ các tầng trung gian của visual encoder trong CLIP
    (ModifiedResNet hoặc VisualTransformer) để tạo perturbation (nhiễu)
    và áp dụng vào các đầu vào của các tầng đó trong quá trình forward pass.
    """
    def __init__(self, with_input_norm=True):
        self._init_inbatch_properties()
        self.with_input_norm = with_input_norm

        self.perturbations = []  # Lưu các gradient thu được
        self.remove_handles = []  # Lưu các handle của hook để gỡ bỏ sau khi dùng

    def _init_inbatch_properties(self):
        # Các danh sách lưu trữ thông tin cần thiết cho việc áp dụng perturbation
        self.perturbation_layers = []          # Chỉ số của các layer cần áp dụng perturbation
        self.perturbation_factor = []          # Hệ số scale cho từng perturbation
        self.perturbation_idices = []          # Chỉ số mẫu (trong toàn dataset)
        self.perturbation_idices_inbatch = []  # Chỉ số mẫu trong batch hiện hành

    def feature_augmentation(self, model, samples, targets, net_type):
        """
        Thực hiện forward pass (trên một số mẫu được chọn) để thu thập gradient từ các tầng trung gian
        của visual encoder. Các gradient này sẽ được sử dụng làm perturbation cho các forward pass sau.
        
        Args:
            model: Mô hình CLIP (toàn bộ).
            samples: Ảnh đầu vào (tensor).
            targets: Nhãn tương ứng.
            net_type: Kiểu mạng của visual encoder. 
                      Sử dụng "resnet" cho ModifiedResNet, "vit" cho VisualTransformer.
        """
        if net_type == "resnet":
            num_layers = 5  # layer1, layer2, layer3, layer4, attnpool
            register_func = register_forward_prehook_modified_resnet
        elif net_type == "vit":
            # Số tầng bằng số residual block trong transformer
            num_layers = len(model.visual.transformer.resblocks)
            register_func = register_forward_prehook_vit
        else:
            raise ValueError(f"Unknown net_type {net_type}.")
        
        self.get_feature_augmentation(model, model.visual, samples, targets, num_layers, register_func)

    def register_perturb_forward_prehook(self, model, net_type):
        """
        Đăng ký forward pre-hook để trong quá trình forward pass,
        áp dụng perturbation (nhiễu) lên đầu vào của các tầng trung gian.
        
        Args:
            model: Mô hình CLIP.
            net_type: Kiểu visual encoder ("resnet" hoặc "vit").
        """
        if net_type == "resnet":
            num_layers = 5
            register_func = register_forward_prehook_modified_resnet
        elif net_type == "vit":
            num_layers = len(model.visual.transformer.resblocks)
            register_func = register_forward_prehook_vit
        else:
            raise ValueError(f"Unknown net_type {net_type}.")
        
        self.register_perturb_forward_prehook_layers(model, model.visual, num_layers, register_func)

    def get_feature_augmentation(self, model, visual_encoder, samples, targets, num_layers, register_func):
        """
        Thu thập gradient của đầu vào các tầng trung gian thông qua forward pre-hook.
        Sau đó, tính toán loss và thực hiện backward để lấy gradient, lưu vào self.perturbations.
        
        Args:
            model: Mô hình CLIP.
            visual_encoder: Thành phần visual encoder (model.visual).
            samples: Ảnh đầu vào.
            targets: Nhãn của ảnh.
            num_layers: Số lượng layer cần thu thập.
            register_func: Hàm đăng ký hook tương ứng với kiến trúc.
        """
        layer_inputs = []

        def get_input_prehook(module, inp):
            inp[0].retain_grad()
            layer_inputs.append(inp[0])
        
        remove_handles = register_func(model, visual_encoder, [get_input_prehook] * num_layers)
        samples.requires_grad_()
        model.eval()
        
        outputs = model(samples)
        # Xử lý output (có thể là tensor, tuple, list hay dict chứa logits)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            logits = outputs.get('logits', None)
            if logits is None:
                raise ValueError("Output dict does not contain 'logits'.")
        else:
            logits = outputs

        cls_loss = F.cross_entropy(logits, targets)
        model.zero_grad()
        cls_loss.backward()

        inp_grads = [inp.grad.detach().clone() for inp in layer_inputs]
        samples.requires_grad_(False)
        self.perturbations = inp_grads

        # Kiểm tra NaN
        for p in self.perturbations:
            if torch.isnan(p).any():
                raise ValueError("NaN detected in gradients")

        for handle in remove_handles:
            handle.remove()

    def register_perturb_forward_prehook_layers(self, model, visual_encoder, num_layers, register_func):
        """
        Đăng ký các hook nhằm cộng perturbation vào đầu vào của các tầng trung gian trong quá trình forward pass.
        
        Args:
            model: Mô hình CLIP.
            visual_encoder: Visual encoder (model.visual).
            num_layers: Số layer cần đăng ký.
            register_func: Hàm đăng ký hook.
        """
        def perturb_input_prehook_full(module: torch.nn.Module, inp, layer_id):
            if layer_id in self.perturbation_layers:
                inp0 = inp[0].clone()
                p_layers = np.array(self.perturbation_layers)
                p_factor = np.array(self.perturbation_factor)
                p_idices = np.array(self.perturbation_idices)
                p_idices_inbatch = np.array(self.perturbation_idices_inbatch)
                p_index = np.nonzero(p_layers == layer_id)[0]
                num_new_axes = len(self.perturbations[layer_id].size()) - 1
                if self.with_input_norm:
                    norm_val = inp0.data[p_idices_inbatch[p_index]].view(len(p_index), -1).norm(dim=-1) ** 2
                    # Thêm các chiều mới vào norm_val bằng cách dùng unsqueeze trong vòng lặp
                    for _ in range(num_new_axes):
                        norm_val = norm_val.unsqueeze(-1)
                    # Chuyển p_factor thành tensor và thêm các chiều mới tương tự
                    factor_tensor = torch.from_numpy(p_factor[p_index]).float()
                    for _ in range(num_new_axes):
                        factor_tensor = factor_tensor.unsqueeze(-1)
                    perturb = norm_val * self.perturbations[layer_id][p_idices[p_index]] * factor_tensor.to(inp0.device)
                else:
                    factor_tensor = torch.from_numpy(p_factor[p_index]).float()
                    for _ in range(num_new_axes):
                        factor_tensor = factor_tensor.unsqueeze(-1)
                    perturb = self.perturbations[layer_id][p_idices[p_index]] * factor_tensor.to(inp0.device)
                inp0[p_idices_inbatch[p_index]] += perturb
                return (inp0,)


        hooks = [partial(perturb_input_prehook_full, layer_id=i) for i in range(num_layers)]
        self.remove_handles.extend(register_func(model, visual_encoder, hooks))


def register_forward_prehook_modified_resnet(model, visual_encoder, hooks):
    """
    Đăng ký forward pre-hook cho visual encoder kiểu ModifiedResNet.
    Giả sử visual_encoder có các tầng: layer1, layer2, layer3, layer4, attnpool.
    """
    remove_handles = []
    remove_handles.append(visual_encoder.layer1.register_forward_pre_hook(hooks[0]))
    remove_handles.append(visual_encoder.layer2.register_forward_pre_hook(hooks[1]))
    remove_handles.append(visual_encoder.layer3.register_forward_pre_hook(hooks[2]))
    remove_handles.append(visual_encoder.layer4.register_forward_pre_hook(hooks[3]))
    remove_handles.append(visual_encoder.attnpool.register_forward_pre_hook(hooks[4]))
    return remove_handles


def register_forward_prehook_vit(model, visual_encoder, hooks):
    """
    Đăng ký forward pre-hook cho visual encoder kiểu VisualTransformer.
    Ở đây, chúng ta đăng ký hook trên từng residual block trong visual_encoder.transformer.resblocks.
    """
    remove_handles = []
    for i, block in enumerate(visual_encoder.transformer.resblocks):
        remove_handles.append(block.register_forward_pre_hook(hooks[i]))
    return remove_handles
