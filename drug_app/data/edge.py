import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re
import time


def generate_edges(input_file="data/drugs.csv", output_file="data/edges.csv", sample_size=None):
    """优化后的边生成函数，支持抽样处理"""
    start_time = time.time()

    # 1. 加载数据
    df = pd.read_csv(input_file)
    print(f"成功加载数据，共 {len(df)} 条药品记录")
    if sample_size:
        df = df.iloc[:sample_size]
        print(f"调试模式：仅处理前 {sample_size} 条记录")

    # 2. 初始化边集合
    edges = []

    # 3. 生成药品-疾病边
    print("正在处理药品-疾病关系...")
    disease_id_counter = 10000
    disease_name_to_id = {}

    for idx, row in df.iterrows():
        indications = re.split(r'[,;|/]', str(row['indication']))
        indications = [i.strip() for i in indications if i.strip()]

        for disease in indications:
            if disease not in disease_name_to_id:
                disease_name_to_id[disease] = disease_id_counter
                disease_id_counter += 1

            edges.append({
                "source": row['drug_id'],
                "target": disease_name_to_id[disease],
                "relation": "treats",
                "weight": 1.0
            })

    # 4. 生成药品-公司边
    print("正在处理药品-公司关系...")
    company_id_counter = 20000
    company_name_to_id = {}

    for _, row in df.iterrows():
        company = str(row['company']).strip()
        if not company:
            continue

        if company not in company_name_to_id:
            company_name_to_id[company] = company_id_counter
            company_id_counter += 1

        edges.append({
            "source": row['drug_id'],
            "target": company_name_to_id[company],
            "relation": "produced_by",
            "weight": 1.0
        })

    # 5. 生成药品相似边（优化版）
    print("正在计算药品相似度...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    desc_matrix = tfidf.fit_transform(df['description'].fillna(''))

    # 使用稀疏矩阵和分批计算
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        print(f"处理批次: {i}-{batch_end}")

        sim_batch = cosine_similarity(csr_matrix(desc_matrix[i:batch_end]), csr_matrix(desc_matrix))
        for j in range(sim_batch.shape[0]):
            for k in range(j + 1, sim_batch.shape[1]):
                if sim_batch[j, k] > 0.6:
                    edges.append({
                        "source": df.iloc[i + j]['drug_id'],
                        "target": df.iloc[k]['drug_id'],
                        "relation": "similar_to",
                        "weight": round(sim_batch[j, k], 4)
                    })

    # 6. 保存结果
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(output_file, index=False)
    print(f"边生成完成，耗时 {time.time() - start_time:.2f} 秒")

    return pd.DataFrame([
                            {"node_id": id_, "name": name, "type": "disease"}
                            for name, id_ in disease_name_to_id.items()
                        ] + [
                            {"node_id": id_, "name": name, "type": "company"}
                            for name, id_ in company_name_to_id.items()
                        ])


if __name__ == "__main__":
    # # 先测试小样本（调试时使用）
    # test_nodes = generate_edges(sample_size=500)

    # 确认无误后再处理全量数据TF-IDF
    full_nodes = generate_edges()