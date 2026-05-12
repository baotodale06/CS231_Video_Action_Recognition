import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# khai báo matrix đồ thị
class Graph():
    def __init__(self):
        self.num_node = 33
        self.edges = [
            # mặt
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10),
            # vai, hông
            (11,12), (11,23), (12,24), (23,24),
            # tay trái
            (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),
            # tay phải
            (12,14), (14,16), (16,18), (16,20), (16,22), (18,20),
            # chân trái
            (23,25), (25,27), (27,29), (29,31), (27,31),
            # chân phải
            (24,26), (26,28), (28,30), (30,32), (28,32)
        ]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        # tạo matrix kề A kích thước 33, 33 giá trị = 0
        A = np.zeros((self.num_node, self.num_node))
        
        # điền số 1 vào các vị trí có nối xương
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
            
        # Nối mỗi khớp với chính nó (Self-loops)
        for i in range(self.num_node):
            A[i, i] = 1
            
        # Chuẩn hóa matrix
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        A_normalized = np.dot(np.dot(D, A), D)
        
        return torch.tensor(A_normalized, dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5):
        super(STGCNBlock, self).__init__()
        
        # Spatial Graph Convolution (tích chập kgian)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal Convolution (tích chập tgian)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
        # Residual connection (tránh nghẽn mạng)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = lambda x: x
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # x dạng: N, C, T, V
        res = self.residual(x)
        
        # nhân matrix tính toán đặc trưng không gian: X * A
        x = self.gcn(x)
        # định dạng lại x để nhân với A (N, C, T, V)
        n, c, t, v = x.size()
        x = x.view(n, c * t, v)
        x = torch.matmul(x, A)
        x = x.view(n, c, t, v)
        
        # tích chập tgian
        x = self.tcn(x)
        
        # cộng residual và kích hoạt
        return self.relu(x + res)


# ST-GCN hoàn chỉnh
class STGCN(nn.Module):
    def __init__(self, num_class=51, in_channels=3, edge_importance_weighting=True):
        super(STGCN, self).__init__()
        
        # tạo đồ thị
        self.graph = Graph()
        # Đưa A vào buffer để PyTorch tự động quản lý
        self.register_buffer('A', self.graph.A)
        
        # mạng lưới các block ST-GCN
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, kernel_size=9, stride=1),
            STGCNBlock(64, 64, kernel_size=9, stride=1),
            STGCNBlock(64, 64, kernel_size=9, stride=1),
            STGCNBlock(64, 128, kernel_size=9, stride=2),
            STGCNBlock(128, 128, kernel_size=9, stride=1),
            STGCNBlock(128, 128, kernel_size=9, stride=1),
            STGCNBlock(128, 256, kernel_size=9, stride=2),
            STGCNBlock(256, 256, kernel_size=9, stride=1),
            STGCNBlock(256, 256, kernel_size=9, stride=1),
        ])
        
        # trọng số quan trọng
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.graph.num_node, self.graph.num_node))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            
        # Lớp phân loại
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # bỏ chiều M để dễ tính hơn
        x = x.squeeze(-1) 
        
        # chuẩn hóa
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous() # Trả về (N, C, T, V)
        
        # chạy qua các layer st-gcn
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
            
        # gộp avg kgian và tgian
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        
        # phân loại
        x = self.fc(x)
        return x