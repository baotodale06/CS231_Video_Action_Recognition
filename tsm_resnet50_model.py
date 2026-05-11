import torch
import torch.nn as nn
import torchvision.models as models

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=16, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment # Số lượng frames trong 1 video (của bạn là 16)
        self.n_div = n_div # Tỉ lệ channel dùng để dịch chuyển (thường là 1/8)

    def forward(self, x):
        # x có shape: [Batch * Time, Channel, H, W]
        bt, c, h, w = x.size()
        t = self.n_segment
        b = bt // t
        x = x.view(b, t, c, h, w) # Tách ra thành [B, T, C, H, W]

        fold = c // self.n_div
        out = torch.zeros_like(x)
        
        # Cơ chế Shift: Dịch khung hình về trước và sau để trao đổi thông tin
        out[:, :-1, :fold] = x[:, 1:, :fold]         # Shift sang trái (Past)
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold] # Shift sang phải (Future)
        out[:, :, 2*fold:] = x[:, :, 2*fold:]        # Giữ nguyên phần còn lại
        
        return self.net(out.view(bt, c, h, w))

def make_tsm_resnet50(num_classes, n_segment=16):
    # Dùng ResNet-50 pre-trained để học nhanh hơn
    base_model = models.resnet50(weights='IMAGENET1K_V1')
    
    # "Phẫu thuật" chèn TSM vào các khối Bottleneck của ResNet
    def add_tsm(model):
        for name, child in model.named_children():
            if isinstance(child, models.resnet.Bottleneck):
                child.conv1 = TemporalShift(child.conv1, n_segment=n_segment)
            else:
                add_tsm(child)
    
    add_tsm(base_model)
    
    # Thay đổi lớp phân loại cuối cùng cho 51 lớp của HMDB51
    base_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(base_model.fc.in_features, num_classes)
    )
    return base_model

class TSM_Network(nn.Module):
    def __init__(self, num_classes=51, n_segment=16):
        super().__init__()
        # Khởi taạo backbone ResNet-50 đã được chèn các lớp Temporal Shift
        self.model = make_tsm_resnet50(num_classes, n_segment)

    def forward(self, x):
        # x shape: [Batch, Time, Channel, H, W] -> (Ví dụ: [4, 16, 3, 224, 224])
        b, t, c, h, w = x.shape
        
        # Bước 1: Ép Batch và Time lại để ResNet 2D có thể đọc được
        # New shape: [Batch * Time, Channel, H, W] -> ([64, 3, 224, 224])
        x = x.view(b * t, c, h, w)
        
        # Bước 2: Chạy qua mạng ResNet-50 (đã có TSM trao đổi thông tin giữa các frames)
        logits = self.model(x) # Output shape: [64, 51] (51 là số lớp hành động)
        
        # Bước 3: Tách lại Batch và Time, sau đó lấy trung bình theo chiều Time (dim=1)
        # logits.view(b, t, -1) -> [4, 16, 51]
        # .mean(dim=1) -> [4, 51] (Kết quả cuối cùng cho 4 video)
        return logits.view(b, t, -1).mean(dim=1)
