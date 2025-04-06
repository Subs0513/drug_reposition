from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .graph.data_loader import load_graph_data
from .graph.gcn import DrugGCN
import torch


def drug_search(request):
    if request.method == 'POST':
        drug_id = request.POST.get('drug_id')

        # 加载图数据
        data = load_graph_data()

        # 初始化模型
        model = DrugGCN(input_dim=data.x.size(1))
        model.load_state_dict(torch.load('drug_app/data/processed/model.pth'))

        # 获取嵌入
        with torch.no_grad():
            embeddings = model(data)

        # 返回相似药品（示例逻辑）
        similar_drugs = find_similar_drugs(embeddings, drug_id)

        return render(request, 'drug/results.html', {'drugs': similar_drugs})

    return render(request, 'drug/search.html')
