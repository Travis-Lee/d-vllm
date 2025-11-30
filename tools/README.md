model.layers.0.self_attn.k_norm.weight torch.Size([128])
model.layers.0.self_attn.k_proj.weight torch.Size([1024, 1024])
model.layers.0.self_attn.o_proj.weight torch.Size([1024, 2048])
model.layers.0.self_attn.q_norm.weight torch.Size([128])
model.layers.0.self_attn.q_proj.weight torch.Size([2048, 1024])
model.layers.0.self_attn.v_proj.weight torch.Size([1024, 1024])
model.layers.1.input_layernorm.weight torch.Size([1024])
model.layers.1.mlp.down_proj.weight torch.Size([1024, 3072])
model.layers.1.mlp.gate_proj.weight torch.Size([3072, 1024])
model.layers.1.mlp.up_proj.weight torch.Size([3072, 1024])
model.layers.1.post_attention_layernorm.weight torch.Size([1024])
model.layers.1.self_attn.k_norm.weight torch.Size([128])
model.layers.1.self_attn.k_proj.weight torch.Size([1024, 1024])
model.layers.1.self_attn.o_proj.weight torch.Size([1024, 2048])
model.layers.1.self_attn.q_norm.weight torch.Size([128])
model.layers.1.self_attn.q_proj.weight torch.Size([2048, 1024])
model.layers.1.self_attn.v_proj.weight torch.Size([1024, 1024])


📊 对应关系表
权重名称	                作用	输出形状
q_proj.weight [2048,1024]	Q 投影	[B,L,2048]
k_proj.weight [1024,1024]	K 投影	[B,L,1024]
v_proj.weight [1024,1024]	V 投影	[B,L,1024]
合并后总和	Q+K+V	[B,L,4096]


总结表
步骤	    张量	        形状
输入	    hidden_states	[2,128,1024]
QKV 投影	qkv	            [2,128,4096]
拆分	    q/k/v	        [2,128,2048]/[2,128,1024]/[2,128,1024]
裁剪	    q/k/v	        [2,128,1024]/[2,128,512]/[2,128,512]
reshape	    q/k/v	        [2,128,16,64]/[2,128,8,64]/[2,128,8,64]
转置	    q/k/v	        [2,16,128,64]/[2,8,128,64]/[2,8,128,64]
RoPE	    q/k	            不变
GQA 扩展	k/v	            [2,16,128,64]/[2,16,128,64]
注意力输出	out	            [2,16,128,64]
合并多头	out	            [2,128,1024]
残差加回	hidden_states	[2,128,1024]


reshape 成多头结构
'''cpp
q = q.view(B, L, num_heads, head_dim)   # [2,128,16,64]
k = k.view(B, L, num_kv_heads, head_dim) # [2,128,8,64]
v = v.view(B, L, num_kv_heads, head_dim) # [2,128,8,64]
'''


📊 最终输出形状
张量	形状	                含义
Q	    [2, 128, 16, 64]	    每个 token 有 16 个 Query head，每个 head 64 维
K	    [2, 128, 8, 64]	        每个 token 有 8 个 Key head，每个 head 64 维
V	    [2, 128, 8, 64]	        每个 token 有 8 个 Value head，每个 head 64 维


## 🧩 Q/K/V 的数值范围
- **均值 (mean)**  
  - 一般接近 **0**（因为权重初始化和归一化都会让分布居中）。  
  - 如果均值特别大（比如 >10），说明数值可能爆炸。

- **标准差 (std)**  
  - 通常在 **0.5 ~ 2** 左右比较合理。  
  - 如果 std ≈ 0，说明张量几乎全是常数 → 注意力退化。  
  - 如果 std >> 10，说明梯度或数值可能爆炸。

- **最小值 / 最大值 (min/max)**  
  - 正常情况下在 **[-10, 10]** 范围内。  
  - 如果出现极端值（比如 ±1e5），说明数值不稳定。


## 🧩 scores (Q·Kᵀ/√d) 的范围
- **均值 (mean)**  
  - 一般接近 **0**。  
- **标准差 (std)**  
  - 通常在 **1 左右**（因为缩放因子 1/√d 控制了方差）。  
- **最小值 / 最大值 (min/max)**  
  - 常见在 **[-5, 5]** 或稍大。  
  - 如果范围特别极端（比如 min=-1000, max=1000），softmax 会变得非常尖锐，只看一个 token → 注意力失效。


## 📊 正常 vs 异常对比表

| 张量              | 正常范围       | 异常情况 |
|------             |-----------     |-----------|
| Q/K/V mean        | ≈ 0            | >> 10 或 << -10 |
| Q/K/V std         | 0.5 ~ 2        | ≈ 0（退化）或 >> 10（爆炸） |
| Q/K/V min/max     | [-10, 10]      | 极端值 ±1e5 |
| scores mean       | ≈ 0            | 偏离过大 |
| scores std        | ≈ 1            | ≈ 0 或 >> 10 |
| scores min/max    | [-5, 5] 或稍大 | 极端值 ±1000 |

---

## ✅ 总结
- **正常情况**：均值接近 0，标准差在 0.5~2，min/max 在 [-10,10] 或稍大，scores 在 [-5,5] 左右。  
- **异常情况**：均值/方差过大或过小，min/max 极端，scores 范围过宽导致 softmax 失效。  


## MPS上的分块计算

分块计算注意力 → 得到每个 chunk 的输出。
拼接所有 chunk → 恢复完整序列。
调整维度 → 从 [B, num_heads, L, head_dim] 转成 [B, L, hidden_size]。
线性投影 → 得到最终的注意力输出，供后续 Transformer Block 使用。

torch.nn.functional.linear(hidden_states, self.lm_head.weight)
'''shell
hidden_states 形状通常是 [B, L, H]：
    B = batch size
    L = 序列长度
    H = hidden_size（比如 1024）

self.lm_head.weight 形状是 [V, H]：
    V = vocab_size（词表大小，比如 50k）
    H = hidden_size

torch.nn.functional.linear(x, W) 本质就是：

𝑦=𝑥⋅𝑊^𝑇
输入 x:[B, L, H]
权重 w:[V, H]
输出 y:[B, L, V]

也就是说，每个位置的隐藏向量都会和词表里的所有词向量做点积，得到一个长度为 V 的分数向量（logits）。

hidden_states [B, L, H]
   ↓ 线性变换 (矩阵乘法)
lm_head.weight [V, H]
   ↓
logits [B, L, V]
'''
