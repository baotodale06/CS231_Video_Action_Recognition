import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# khai báo đồ thị 
class Graph():
    def __init__(self):
        self.num_node = 33
        self.edges = [
            # đầu, mặt
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10),
            # thân
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
        # lấy điểm 0 (mũi) làm trọng tâm để tính khoảng cách hướng tâm/ly tâm
        self.center = 0 
        self.A = self.get_spatial_graph()

    def get_spatial_graph(self):
        # tạo ma trận kề cơ bản
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
            
        # tính khoảng cách từ mỗi node tới node 0 (center)
        distance_to_center = np.full(self.num_node, np.inf)
        distance_to_center[self.center] = 0
        visited = [self.center]
        queue = [self.center]
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(self.num_node):
                if A[node, neighbor] == 1 and neighbor not in visited:
                    distance_to_center[neighbor] = distance_to_center[node] + 1
                    visited.append(neighbor)
                    queue.append(neighbor)

        # phân hoạch làm 3 ma trận: 0 (local), 1 (hướng tâm), 2 (ly tâm)
        A_split = np.zeros((3, self.num_node, self.num_node))
        for i in range(self.num_node):
            for j in range(self.num_node):
                if A[i, j] == 1 or i == j:
                    if distance_to_center[i] == distance_to_center[j]:
                        A_split[0, i, j] = 1
                    elif distance_to_center[i] < distance_to_center[j]:
                        A_split[1, i, j] = 1
                    else:
                        A_split[2, i, j] = 1

        # chuẩn hóa ma trận
        for k in range(3):
            D = np.sum(A_split[k], axis=1)
            D[D == 0] = 1
            D_inv = np.diag(D ** -0.5)
            A_split[k] = np.dot(np.dot(D_inv, A_split[k]), D_inv)
            
        return torch.tensor(A_split, dtype=torch.float32)

# tích chập đồ thị không gian
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel=3):
        super().__init__()
        self.s_kernel = s_kernel
        self.conv = nn.Conv2d(in_channels, out_channels * s_kernel, kernel_size=1)

    def forward(self, x, A):
        # x: (N, C, T, V)
        # A: (3, V, V)
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel, kc // self.s_kernel, t, v)
        
        # áp dụng công thức ST-GCN gốc: Nhân với ma trận 3D A
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5):
        super().__init__()
        
        # Spatial Conv (tchập kgian)
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.tcn_bn1 = nn.BatchNorm2d(out_channels)
        
        # Temporal Conv (tchập tgian)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout) # Thêm dropout chống overfit
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
        res = self.residual(x)
        x = self.tcn_bn1(self.gcn(x, A))
        x = self.tcn(x)
        return self.relu(x + res)

# ST-GCN hoàn chỉnh
class STGCN(nn.Module):
    def __init__(self, num_class=51, in_channels=3, dropout=0.5):
        super().__init__()
        
        self.graph = Graph()
        self.register_buffer('A', self.graph.A)
        
        # chuẩn hóa
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        
        # mạng lưới các block ST-GCN
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, kernel_size=9, stride=1, dropout=dropout),
            STGCNBlock(64, 64, kernel_size=9, stride=1, dropout=dropout),
            STGCNBlock(64, 64, kernel_size=9, stride=1, dropout=dropout),
            STGCNBlock(64, 128, kernel_size=9, stride=2, dropout=dropout),
            STGCNBlock(128, 128, kernel_size=9, stride=1, dropout=dropout),
            STGCNBlock(128, 128, kernel_size=9, stride=1, dropout=dropout),
            STGCNBlock(128, 256, kernel_size=9, stride=2, dropout=dropout),
            STGCNBlock(256, 256, kernel_size=9, stride=1, dropout=dropout),
            STGCNBlock(256, 256, kernel_size=9, stride=1, dropout=dropout),
        ])
        
        # trọng số quan trọng
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.graph.A.size()))
            for _ in self.st_gcn_networks
        ])
            
        self.fc = nn.Linear(256, num_class)
        self.final_dropout = nn.Dropout(p=dropout) # chống overfit trước khi dự đoán

    def forward(self, x):
        # bỏ chiều M
        x = x.squeeze(-1) 
        
        # chuẩn hóa
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous() # trả về (N, C, T, V)
        
        # Chạy qua các layer ST-GCN
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
            
        # global pooling (avg kgian & tgian)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        
        # phân loại
        x = self.final_dropout(x)
        x = self.fc(x)
        return x