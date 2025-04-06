import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载你的原始数据
df = pd.read_csv("data/drugs.csv")

# 生成药品节点
tfidf = TfidfVectorizer(max_features=50)
desc_features = tfidf.fit_transform(df['description'])

drug_nodes = pd.DataFrame({
    'node_id': df['drug_id'],
    'name': df['name'],
    'type': 'drugs',
    'features': [x.tolist()[0] for x in desc_features.toarray()]
})

# 从适应症字段提取疾病节点
diseases = set()
for ind in df['indication']:
    diseases.update(ind.split(';'))  # 假设适应症用分号分隔

disease_nodes = pd.DataFrame({
    'node_id': [1000+i for i in range(len(diseases))],
    'name': list(diseases),
    'type': 'disease',
    'features': [[0]*50 for _ in diseases]  # 临时填充
})

nodes = pd.concat([drug_nodes, disease_nodes])
nodes.to_csv("data/nodes.csv", index=False)