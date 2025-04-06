import torch
from torch_geometric.data import Data
import pandas as pd


def load_graph_data():
    """从CSV加载并转换为PyG图数据"""
    nodes = pd.read_csv('drug_app/data/raw/nodes.csv')
    edges = pd.read_csv('drug_app/data/raw/edges.csv')

    # 节点特征
    x = torch.tensor(nodes['features'].apply(eval).tolist(), dtype=torch.float)

    # 边索引
    edge_index = torch.tensor([edges['source'], edges['target']], dtype=torch.long)

    # 构建图数据
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=torch.tensor(edges['weight'].values) if 'weight' in edges else None
    )

    # 保存处理后的数据
    torch.save(data, 'drug_app/data/processed/graph_data.pt')
    return data