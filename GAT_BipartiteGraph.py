import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

#################################
# 配置区：在此修改BERT模型路径 ↓↓↓
#################################
BERT_PATH = "/home/iip/Documents/2024050075/GAT/model"  # 你的本地路径（确保此路径存在）

# 验证模型文件是否存在
assert Path(BERT_PATH).exists(), f"BERT模型路径{BERT_PATH}不存在！"
assert (Path(BERT_PATH)/"pytorch_model.bin").is_file(), "缺少模型权重文件pytorch_model.bin"
assert (Path(BERT_PATH)/"config.json").is_file(), "缺少配置文件config.json"
assert (Path(BERT_PATH)/"vocab.txt").is_file(), "缺少词表文件vocab.txt"

#################################
# 中文文本编码模块
#################################
class ChineseBERTEncoder(nn.Module):
    def __init__(self, model_path):  # 接收模型路径
        super().__init__()
        print(f"正在从本地加载BERT模型: {model_path}...")
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # 冻结BERT参数以加速训练
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        # 自动检测GPU可用性
        device = next(self.bert.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # 取CLS向量

#################################
# 二分图构建模块
#################################
class BipartiteGraph:
    def __init__(self, problems, concepts, relations, bert_path):
        """
        参数说明:
        - problems: 字典 {问题ID: 问题文本}
        - concepts: 字典 {概念ID: 概念文本}
        - relations: 列表 [(问题ID, 概念ID), ...] 
        - bert_path: BERT模型的本地路径
        """
        # 初始化编码器
        self.encoder = ChineseBERTEncoder(bert_path)
        
        # 节点和边初始化
        self.problem_ids = list(problems.keys())
        self.concept_ids = list(concepts.keys())
        
        # 构建显式边（问题->概念）
        explicit_src = [self.problem_ids.index(p) for p, _ in relations]
        explicit_dst = [len(self.problem_ids) + self.concept_ids.index(c) for _, c in relations]
        
        # 构建隐式边（示例为全连接问题-问题）
        num_problems = len(self.problem_ids)
        src = torch.arange(num_problems).repeat_interleave(num_problems)
        dst = torch.arange(num_problems).repeat(num_problems)
        implicit_edges = torch.stack([src, dst], dim=0)
        
        # 合并边索引
        self.edge_index = torch.cat([
            torch.tensor([explicit_src, explicit_dst], dtype=torch.long),
            implicit_edges
        ], dim=1)
        
        # 生成节点特征
        with torch.no_grad():
            self.x_problem = self.encoder([problems[p] for p in self.problem_ids])
            self.x_concept = self.encoder([concepts[c] for c in self.concept_ids])
        
        self.x = torch.cat([self.x_problem, self.x_concept], dim=0)
    
    def get_graph_data(self):
        return Data(x=self.x, edge_index=self.edge_index)

#################################
# 图神经网络模块
#################################
class ConceptGAT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=3, concat=True)
        self.gat2 = GATConv(hidden_dim*3, hidden_dim, heads=1)
        
    def forward(self, data):
        x = F.elu(self.gat1(data.x, data.edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, data.edge_index)
        return x

#################################
# 示例使用
#################################
if __name__ == "__main__":
     # 加载本地数据集
    df_problems = pd.read_csv("/home/iip/Documents/2024050075/GAT/data/problem.csv", encoding='UTF-8')  # 根据实际编码调整
    your_problems = dict(zip(df_problems['id'], df_problems['title']))
    
    df_concepts = pd.read_csv("/home/iip/Documents/2024050075/GAT/data/problem_tag.csv")
    your_concepts = dict(zip(df_concepts['id'], df_concepts['name']))
    
    df_relations = pd.read_csv("/home/iip/Documents/2024050075/GAT/data/problem_tags.csv")
    your_relations = [ (row['problem_id'], row['problemtag_id']) for _, row in df_relations.iterrows() ]
    
    # 数据完整性校验
    assert all(p in your_problems for p, _ in your_relations), "存在未知问题ID!"
    assert all(c in your_concepts for _, c in your_relations), "存在未知概念ID!"
    
    # 构建图（其余代码不变）
    graph = BipartiteGraph(
        problems=your_problems,
        concepts=your_concepts,
        relations=your_relations,
        bert_path=BERT_PATH
    )
    graph_data = graph.get_graph_data()
    
    # 初始化模型
    model = ConceptGAT()
   # 前向计算
    embeddings = model(graph_data)
    problem_emb = embeddings[:len(your_problems)]
    concept_emb = embeddings[len(your_problems):]
    
    print("\n====== 运行结果 ======")
    print(f"总节点数: {graph_data.num_nodes}  总边数: {graph_data.edge_index.shape[1]}")
    print(f"问题嵌入形状: {problem_emb.shape}")
    print(f"概念嵌入形状: {concept_emb.shape}")
    print("======================")
