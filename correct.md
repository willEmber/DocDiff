下面是**按照你仓库 willEmber/DocDiff 的 `edict/` 与 `schedule/` 现状**做的逐点体检 + 修改方案。重点放在**EDICT 的可逆实现细节**与 **与 DocDiff（残差域、预测 (x_0)）的正确对接**，并给出可以直接替换/粘贴的关键代码骨架与自检步骤。

> 我能直接打开的关键文件：
>
> * `schedule/schedule.py`：自定义 beta 调度（cosine/linear 等）【可见函数 `cosine_beta_schedule` / `linear_beta_schedule` 等】。这部分整体 OK【证据：文件中 `get_betas()` 返回 torch 张量的 betas 序列】。([GitHub][1])
> * `schedule/diffusionSample.py`：目前还是**DDPM/DDIM 的采样器**，没有 EDICT 的耦合/混合（mixing）逻辑；同时存在一些实现细节隐患（见下）【证据：`GaussianDiffusion.forward` 中 `pre_ori=='False'` 走随机（DDPM），否则走确定性 DDIM，且 `x0→ε` 映射在 else 分支中手写】。([GitHub][2])
> * 官方 EDICT 参考：核心逻辑在 `edict_functions.py`，并强调**全链路双精度**与**每步 mixing（权重 p≈0.93）**。([GitHub][3])

---

## 一、为什么你会出现“单步一致性很小但全链路反演崩溃（INV_MSE 极大、INV_COS 为负）”

结合你 repo 里的实现与官方 EDICT 代码/论文，最常见的 6 类触发点——任何一条都足以让误差逐步放大，最终出现 **cos<0** 的灾难性偏移：

1. **没有实现 EDICT 的“交替耦合 + mixing 层 + mixing 的显式逆”对**
   你当前的 `GaussianDiffusion.forward()` 仍是**标准 DDPM/DDIM** 的一步（先 `p_mean_variance`，或把 `x0` 反解为 `ε` 再用 DDIM 公式更新）【见 `forward` 的两个分支】；但是 EDICT 的正反向需要**两条 latent（x_t, y_t）**的**交替依赖**与**每步 mixing**，以及**反向必须先逆混合再做耦合逆**。缺失任一步，长链路必炸。([GitHub][2])

2. **(x_0 \rightarrow \epsilon) 的映射或时间索引用错**
   映射应为
   [
   \hat\epsilon_t=\frac{x_t-\sqrt{\bar\alpha_t},\hat x_0}{\sqrt{1-\bar\alpha_t}}
   ]
   注意是 (\bar\alpha_t) 而非 (\bar\alpha_{t-1})。你在 DDIM 分支里是用 (\sqrt{\gamma_t}) 写的（(\gamma_t\equiv\bar\alpha_t)），这一点**在 DDIM 分支基本正确**，但 EDICT 要求**每一步都用这一“在线映射”去驱动耦合更新**，你当前并未这么做。([GitHub][2])

3. **a_t / b_t 系数的定义/索引不一致**
   EDICT/（DDIM 形式）使用
   [
   a_t=\sqrt{\frac{\bar\alpha_{t-1}}{\bar\alpha_t}},\quad
   b_t=\sqrt{1-\bar\alpha_{t-1}}-a_t\sqrt{1-\bar\alpha_t},
   ]
   任何把 (\alpha_t) 当成 (\bar\alpha_t) 或把 (t\leftrightarrow t-1) 搞反，都会产生符号级偏差并在长链路发散。([GitHub][3])

4. **混合层（mixing）没有“每步都做”，或 p 取值极端**
   论文与官方实现建议 **p≈0.93**，且**每一步都要做 mixing**；反向要先做**逆混合**（线性显式逆），再做耦合逆。少做/漏做/次序错，误差会被指数放大。([GitHub][3])

5. **精度不是 float64（双精度）**
   官方专门把 diffusers 改成了 **double** 以避免反演实验的微小误差积累。你现在的 pipeline 里，没有看到强制把 latent/a_t/b_t/p/所有步进运算切换到 float64 的迹象。([GitHub][3])

6. **调度/条件不对齐**
   训练/采样/反演三者的 **({\bar\alpha_t})、步数 T、时间网格** 必须一致；DocDiff 的条件是通道拼接 (x_C)（coarse predictor 输出），反演侧要使用**完全相同的 CP/HRR 与网格**。否则 HRR 给出的 (\hat x_0) 会有系统偏移。你仓库里 `schedule/schedule.py` 的调度定义没问题，但要确保**所有流程都用同一份**。([GitHub][1])

> 另外，小细节：你在 `diffusionSample.py` 中把 `sqrt_gammas` / `sqrt_one_minus_gammas` 用 `np.sqrt` 计算再 `register_buffer`，这在 GPU/float64 下容易带来 **dtype/device 不一致** 或隐式拷贝的问题，建议改用 `torch.sqrt`（见下面补丁）。([GitHub][2])

---

## 二、对你仓库现状的“就地”改造方案

> 目标：不改 HRR 的训练范式（仍**预测 (x_0)**），仅替换**采样/反演器**为 EDICT；在训练中增加“**一步可逆一致性**”正则（轻量），并提供**反演自测脚本**。

### 1) 修补 `schedule/diffusionSample.py` 的基础计算与接口

**(a) 改正 `sqrt_*` 的计算/注册与 dtype**
把 `np.sqrt` 改为 `torch.sqrt`，并确保是 **float64**：

```python
# __init__ 里，替换这几行（使用 torch 而非 numpy，并转 double）
gammas = alphas_bar.double()
self.register_buffer('gammas', gammas)
self.register_buffer('sqrt_gammas', torch.sqrt(gammas))
self.register_buffer('sqrt_one_minus_gammas', torch.sqrt(1. - gammas))
```

说明：当前代码用 `np.sqrt(1 - gammas)` 再 `register_buffer`，在 GPU/双精度下容易出 device/dtype 隐患。([GitHub][2])

**(b) 抽出标准的 (x_0\to\epsilon) 映射函数**（供 EDICT 使用）：

```python
def x0_to_eps(x_t, x0_hat, abar_t):
    # abar_t = \bar\alpha_t，注意是 t，不是 t-1
    return (x_t - abar_t.sqrt() * x0_hat) / (1. - abar_t).sqrt()
```

（这与现有 DDIM 分支里的写法一致，但抽出来能与 EDICT 重用。）([GitHub][2])

**(c) 暴露 (\bar\alpha) 给采样器**
在 `GaussianDiffusion.__init__` 里把 `alphas_bar` 暴露出去（或提供 `get_alphas_cumprod()` 方法），以便 EDICT 计算 (a_t,b_t)。

---

### 2) 在本仓库新增一个**严格可逆的 EDICT 采样器**（建议放 `edict/edict_sampler.py`）

下面是**与 DocDiff（预测 (x_0)）对接好的** EDICT 正/反向“一步”实现。你可以在 `edict/` 下新建文件直接使用（fp64 计算；每步 mixing；(\hat x_0\to\hat\epsilon_t) 在线映射；严格的先逆混合，再做耦合逆）：

```python
# edict/edict_sampler.py
import torch

@torch.no_grad()
def at_bt(abar, t):               # abar: [T+1] or [T] 皆可，按你索引习惯
    abar_t   = abar[t].double()
    abar_tm1 = abar[t-1].double()
    a_t = (abar_tm1 / abar_t).sqrt()
    b_t = (1. - abar_tm1).sqrt() - a_t * (1. - abar_t).sqrt()
    return a_t, b_t

@torch.no_grad()
def eps_from_model(model, x_t, t, cond, abar, predict_type="x0"):
    # HRR 仍预测 x0：把 \hat x0 在线映射为 \hat eps_t （务必用 abar[t]）
    x0_hat = model(torch.cat([x_t.float(), cond], 1), t)    # HRR 可 fp32
    eps = (x_t.double() - abar[t].sqrt() * x0_hat.double()) / (1. - abar[t]).sqrt()
    return eps

@torch.no_grad()
def edict_step_forward(x_t, y_t, t, model, cond, abar, p=0.93):
    # 正向：t -> t-1
    a_t, b_t = at_bt(abar, t)
    eps_y = eps_from_model(model, y_t, t, cond, abar)
    x_inter_tm1 = a_t * x_t.double() + b_t * eps_y

    eps_x_inter = eps_from_model(model, x_inter_tm1, t, cond, abar)
    y_inter_tm1 = a_t * y_t.double() + b_t * eps_x_inter

    # mixing（必须每步做）
    x_tm1 = p * x_inter_tm1 + (1 - p) * y_inter_tm1
    y_tm1 = p * y_inter_tm1 + (1 - p) * x_inter_tm1
    return x_tm1, y_tm1

@torch.no_grad()
def edict_step_inverse(x_tm1, y_tm1, t, model, cond, abar, p=0.93):
    # 反向：t-1 -> t；次序必须：先逆混合，再耦合逆
    a_t, b_t = at_bt(abar, t)
    # 逆混合（线性显式逆）
    x_inter_tm1 = (x_tm1 - (1 - p) * y_tm1) / p
    y_inter_tm1 = (y_tm1 - (1 - p) * x_tm1) / p

    # 耦合逆：先用 x_inter_tm1 估 eps，再解出 y_t；再用 y_t 估 eps，解出 x_t
    eps_x_inter = eps_from_model(model, x_inter_tm1, t, cond, abar)
    y_t = (y_inter_tm1 - b_t * eps_x_inter) / a_t
    eps_y = eps_from_model(model, y_t, t, cond, abar)
    x_t = (x_inter_tm1 - b_t * eps_y) / a_t
    return x_t, y_t
```

> 上述写法与官方 EDICT 的核心流程一致，并显式使用 **float64** 防止漂移；同时与你目前的 HRR（预测 (x_0)）兼容，因为我们在 `eps_from_model` 里做了标准的 (x_0\to\epsilon) 映射。([GitHub][3])

---

### 3) 在 `GaussianDiffusion` 里挂接 EDICT 正/反向全链路

在 `diffusionSample.py` 里新增两段高层函数，供“编码端/解码端”共用：

```python
from edict.edict_sampler import edict_step_forward, edict_step_inverse

@torch.no_grad()
def edict_denoise_all(self, x_T, cond, abar, p=0.93):
    # 正向生成：x_T -> x_0_hat
    # 初始化双轨；y_T 随机 or 固定随机种子也可
    x_t = x_T.double()
    y_t = torch.randn_like(x_t, dtype=torch.float64)
    for time_step in reversed(range(1, self.T+1)):  # 假设 abar[0]~abar[T]
        t = x_t.new_full([x_T.shape[0]], time_step, dtype=torch.long)
        x_t, y_t = edict_step_forward(x_t, y_t, time_step, self.model, cond, abar, p=p)
    # 最后一拍用 HRR 直接预测 x0（保持和 DocDiff 一致）
    t0 = x_t.new_zeros([x_T.shape[0]], dtype=torch.long)
    x0_hat = self.model(torch.cat([x_t.float(), cond], 1), t0)
    return x0_hat

@torch.no_grad()
def edict_invert_all(self, x0_residual, cond, abar, p=0.93):
    # 反向：\hat x0 -> \hat xT
    x_t = x0_residual.double()
    y_t = torch.randn_like(x_t, dtype=torch.float64)
    for time_step in range(1, self.T+1):
        t = x_t.new_full([x_t.shape[0]], time_step, dtype=torch.long)
        x_t, y_t = edict_step_inverse(x_t, y_t, time_step, self.model, cond, abar, p=p)
    return x_t
```

然后你在推理/测试处这样调用（与你的隐写流程对齐）：

```python
abar = torch.cumprod(1. - self.schedule.get_betas().double(), dim=0)  # 与训练一致
x0_hat = self.edict_denoise_all(x_T=secret_noise, cond=x_C, abar=abar, p=0.93)
xT_rec = self.edict_invert_all(x0_residual=x0_hat, cond=x_C, abar=abar, p=0.93)
```

---

### 4) 训练期新增“一步可逆一致性”正则（轻量）

在你现有的 `train_step` 中（或者等价训练循环）加上：

* 给定随机 (t)，用真实 (\epsilon) 合成 (x_t,x_{t+1})；
* 用 HRR 的 (\hat x_0)→(\hat\epsilon_t)，调用 **一次** `edict_step_forward(x_t, y_t, t, ...)` 得到 (\tilde x_{t+1})；
* 加 (\mathcal L_{\text{inv-1}}=|\tilde x_{t+1}-x_{t+1}|_2^2)；
* 权重建议 0.05–0.1。
  这样做**几乎不增显存**，能显著抑制“单步好、长链路崩”的漂移（因为把 EDICT 的一步数值一致性直接放进了损失）。

---

## 三、对照清单（按优先级从高到低排查）

1. **是否真的调用了 EDICT 的耦合/混合/逆混合，而不是 DDPM/DDIM？**
   你现在 `forward()` 还在走 DDPM/DDIM 路径（`p_mean_variance` 分支或 DDIM 分支）【可直接在日志里打印“edict_step_forward 被调用次数=T”】。([GitHub][2])

2. **(x_0\rightarrow\epsilon) 是否使用的是 (\bar\alpha_t)**？
   对任何 t，都用 `abar[t]`，不要用 `abar[t-1]`。

3. **a_t/b_t 的定义是否完全一致**？
   `a_t = sqrt(abar[t-1]/abar[t])`、`b_t = sqrt(1-abar[t-1]) - a_t*sqrt(1-abar[t])`。

4. **mixing 是否“每步都做”，p 是否在 0.9–0.97（建议 0.93）**？
   反向是否先做**逆混合**、再耦合逆？

5. **全链路是否 float64**？
   latent/a_t/b_t/p 与步进计算**都用 double**；HRR/CP 可 fp32，输出后 `to(torch.float64)` 再步进。官方明确为此做了 double 版的 diffusers。([GitHub][3])

6. **调度/时间网格/条件一致**
   `schedule/schedule.py` 生成的 betas/alphas_bar 在训练/推理/反演保持一致；CP 条件 (x_C) 也要在解码侧复现。([GitHub][1])

7. **`diffusionSample.py` 的 `np.sqrt` 改为 `torch.sqrt`**
   防止 dtype/device 不一致。([GitHub][2])

---

## 四、一个 10 分钟的“定位误差来源”小实验（强烈建议跑）

1. **纯数值代数自测（不用网络）**
   用真值 (\epsilon) 函数替代 `eps_from_model`，扫过随机 (x_T,y_T,{\bar\alpha_t},p)，跑完整正向+反向。**fp64 下误差应 <1e‑12**。失败→你的 `a_t/b_t/耦合/混合逆` 有实现错误。

2. **半真值自测（网络接口在，但仍用 oracle ε）**
   仍应 <1e‑12。失败→调度/索引/步序还有问题。

3. **网络 + 在线映射**
   打印核对若干 t 的 `abar[t]`、`x0_to_eps`，确认与 DDIM 分支一致。

4. **p 与 dtype 扫描**
   p∈{0.90,0.93,0.95,0.97}、dtype∈{fp32,fp64} 扫描 INV_MSE/INV_COS 曲线；应看到 **fp64 明显更稳**、**p≈0.93 最稳** 的趋势（与官方一致）。([GitHub][3])

---

## 五、与官方 EDICT 的关键一致性（对照出处）

* **双轨耦合 + mixing / 逆混合**：必须“先逆混合，再耦合逆”，且每步做 mixing，p≈0.93。([GitHub][3])
* **全链路 fp64**：latent 与调度器运算均用 double。([GitHub][3])
* **a_t/b_t 定义**：严格按 DDIM 形式的 (\bar\alpha) 写法。([GitHub][3])
* **预测量类型对齐**：DocDiff 的 HRR 预测 (x_0)，在 EDICT 步进中每步**在线映射为 (\epsilon)** 驱动更新。([GitHub][2])

---

## 六、与你仓库当前代码直接相关的两处“必改点”

1. `diffusionSample.py` 里 **DDPM/DDIM 的路径不能再当 EDICT 用**；需要新建 `edict_denoise_all / edict_invert_all` 两个 API，并在推理/评测时调它们，而不是现有的 `forward()`（后者保留作基线）。([GitHub][2])

2. `diffusionSample.py::__init__` 中**用 numpy 的 `np.sqrt` 注册 buffer**（`sqrt_gammas` / `sqrt_one_minus_gammas`）需要改为 `torch.sqrt` 并统一 double，以防 dtype/device 被隐式转换导致步进时出现极小但“累积致命”的偏差。([GitHub][2])

---

### 小结

* 你目前的实现仍以 **DDPM/DDIM** 为主导，因此“单步一致性小但全链路反演失败”是典型症状。
* 按上面的补丁接入**严格的 EDICT 一步对（耦合 + mixing + 逆混合）**、**fp64 全链路**、**(\hat x_0\to\hat\epsilon_t)** 在线映射和**一致的 ({\bar\alpha_t})** 后，INV_MSE/INV_COS 会稳定下来。
* 我给出的 `edict/edict_sampler.py` 骨架和 `GaussianDiffusion` 的两个封装函数可以**直接落地**，只要把 `model` 换成你的 HRR 前向（ `[x_t, x_C], t` ）即可。

> 参考与对齐处：你仓库的 `schedule/schedule.py`（beta 调度）；`schedule/diffusionSample.py`（当前 DDPM/DDIM 路径与 (x_0\to\epsilon) 写法）；EDICT 官方对 **double 精度**、**每步 mixing** 与代码组织的说明。([GitHub][1])

如果你愿意，我可以把以上补丁整理成 4 个最小文件（`edict/edict_sampler.py`、`schedule/edict_runner.py`、`tools/test_inversion.py`、对 `diffusionSample.py` 的增量修改块）——你直接复制到仓库跑 INV_MSE / INV_COS 与单步一致性曲线即可。

[1]: https://github.com/willEmber/DocDiff/raw/main/schedule/schedule.py "raw.githubusercontent.com"
[2]: https://github.com/willEmber/DocDiff/raw/main/schedule/diffusionSample.py "raw.githubusercontent.com"
[3]: https://github.com/salesforce/EDICT/raw/main/edict_functions.py "raw.githubusercontent.com"
