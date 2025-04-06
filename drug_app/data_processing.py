import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据
df = pd.read_csv("data/drugs.csv")

# 处理缺失值
df['description'] = df['description'].fillna('')
df['indication'] = df['indication'].fillna('未分类')

# 合并文本特征
df['text'] = df['description'] + " " + df['indication']

# TF-IDF特征提取
tfidf = TfidfVectorizer(max_features=500)
text_features = tfidf.fit_transform(df['text'])

# 保存处理后的数据
df.to_csv("data/processed_drugs.csv", index=False)