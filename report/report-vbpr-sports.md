VBPR + Sports

Recall@5   = 0.0359
Recall@10  = 0.0559
Recall@20  = 0.0857
Recall@50  = 0.1391

NDCG@5     = 0.0241
NDCG@10    = 0.0306
NDCG@20    = 0.0383
NDCG@50    = 0.0492

Precision@5  = 0.0080
Precision@10 = 0.0063

MAP@5      = 0.0195
MAP@10     = 0.0221

Parameters:['seed', 'reg weight']=(999,2.0)

VBPR 的核心不是“使用了图片”，而是把用户偏好拆成了协同偏好 + 内容/模态偏好。

\(L_{BPR} = -\log\sigma(s(u,i^+) - s(u,i^-))\)
\(L=L_{BPR} + \lambda L_{reg}\)

为了搜索 \lambda， 所以在 reg_weight = \[ 2.0, 1.0, 0.1, 0.01... \]里搜索，最终最优的是2.0