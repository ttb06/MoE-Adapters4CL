import torch
import torch.nn.functional as F

def compute_class_prototypes(model, data_loader, device, taskid):
    """
    Tính prototype cho mỗi lớp từ tập dữ liệu data_loader, dựa trên embedding được trích xuất từ mô hình.
    
    Args:
        model: Mô hình học, cần có hàm forward nhận các đối số (images, texts, taskid, is_train)
        data_loader: DataLoader cho tập dữ liệu của task cần tính prototype
        device: Thiết bị tính toán (ví dụ: torch.device("cuda"))
        taskid: Task id tương ứng để truyền vào hàm forward của model
    
    Returns:
        prototypes: Dictionary chứa prototype cho mỗi lớp (key: label, value: vector prototype)
    """
    model.eval()
    prototypes = {}
    counts = {}
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Gọi forward của model với taskid chính xác và is_train=False để tính embedding
            outputs = model(images, None, taskid=taskid, is_train=False)
            # Giả sử outputs là embedding tensor có shape [batch, dim]
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            for emb, label in zip(embeddings, labels):
                label = label.item()
                if label not in prototypes:
                    prototypes[label] = emb.clone()
                    counts[label] = 1
                else:
                    prototypes[label] += emb
                    counts[label] += 1
    # Tính trung bình embedding cho mỗi lớp
    for label in prototypes:
        prototypes[label] /= counts[label]
    return prototypes

def prototype_distillation_loss(current_embeddings, targets, old_prototypes, device, margin=0.0):
    loss = 0.0
    count = 0
    for emb, label in zip(current_embeddings, targets):
        label = label.item()
        if label in old_prototypes:
            proto = old_prototypes[label].to(device)
            loss += F.mse_loss(emb, proto)
            count += 1
    if count > 0:
        loss /= count
    return loss
