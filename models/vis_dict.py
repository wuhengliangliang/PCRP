import torch
import torch.nn as nn
import torch.distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    """
    对所有 GPU 的 tensor 做 all_gather，然后在 dim=0 拼接起来。

    ⚠️注意：
      - torch.distributed.all_gather 本身不支持梯度（无反传）。
      - 这里加了 @torch.no_grad()，明确告诉你：这个函数只做统计/同步。

    输入:
      tensor: 任意形状张量（通常你这里用的是 (1, K) 或 (1, K, D) 之类）

    输出:
      output: 把 world_size 份 tensor 沿 dim=0 concat 的结果
              shape = (world_size * tensor.shape[0], ...) 
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # 把每张卡收集到的 tensor 拼起来
    output = torch.cat(tensors_gather, dim=0)
    return output


def sum_inplace(sum_data, new):
    """
    原地相加：sum_data += new
    使用 .data 是为了绕过 autograd（这里这些 buffer 本来就不需要梯度）。
    """
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    """
    拉普拉斯平滑（避免出现 0 计数导致除 0）：
      (x + eps) / (sum(x) + n_categories * eps)

    输入:
      x: (K,) 每个 cluster 的计数（或 EMA 计数）
      n_categories: K
      eps: 平滑项

    输出:
      一个“归一化后的分布”，总和为 1
    """
    return (x + eps) / (x.sum() + n_categories * eps)


def ema_tensor_inplace(moving_avg, new, decay):
    """
    EMA 更新（指数滑动平均），原地写回 moving_avg：

      moving_avg = decay * moving_avg + (1 - decay) * new

    注意这里用 new.detach()：
      - 防止 new 的计算图进入 moving_avg（本来就不需要梯度）
      - moving_avg 通常是 buffer / 统计量
    """
    new_out = torch.mul(new, 1.0 - decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())


class VisualDict(nn.Module):
    """
    VisualDict：一个“可学习的视觉字典/码本”（类似 VQ-VAE 的 codebook，但更新方式是 EMA + 分布式同步）

    核心思路：
      1) 对输入特征 inputs_flatten 计算到每个 codebook 向量的距离
      2) 选最近的 code (argmin)
      3) 训练时用 encodings（one-hot）统计每个 code 被选中的次数 + 被分配到的特征和
      4) 用 EMA 更新 cluster_size / embed_avg
      5) 用 embed_avg / cluster_size 得到新的 embed（码本向量）
      6) 用 straight-through estimator (STE) 把量化结果回传梯度到 inputs_flatten

    参数：
      num_tokens = K：码本大小（token 数）
      token_dim  = D：每个 token 向量维度
      decay / max_decay：EMA 衰减因子控制
      eps：拉普拉斯平滑常数
    """
    def __init__(self, num_tokens, token_dim, decay=0.1, max_decay=0.99, eps=1e-5) -> None:
        super().__init__()
        self.num_tokens = num_tokens    # K
        self.token_dim = token_dim      # D
        self.decay = decay
        self.cur_decay = decay          # 当前 EMA 衰减（可动态调大）
        self.max_decay = max_decay
        self.eps = eps

        # -------------------------
        # 码本 embed: (K, D)，作为 buffer（不走 optimizer 梯度更新）
        # -------------------------
        embed = torch.randn(num_tokens, token_dim)
        self.register_buffer("embed", embed)
        nn.init.normal_(self.embed)

        # -------------------------
        # 分布式 / EMA 统计量
        # cluster_size: (K,)   每个 token 被选中的“次数”（EMA 版本）
        # cluster_sum : (K,)   累积被选中的次数（更像 raw sum）
        # embed_avg   : (K, D) 每个 token 对应分配到的特征向量和（EMA 版本）
        # -------------------------
        self.register_buffer("cluster_size", torch.zeros(num_tokens))
        self.register_buffer("cluster_sum", torch.zeros(num_tokens))
        self.register_buffer("embed_avg", torch.zeros(num_tokens, token_dim))

    def set_decay_updates(self, num_update) -> None:
        """
        动态调节 EMA 衰减（越更新越“稳”）：
          cur_decay = min(cur_decay * num_update, max_decay)

        例如 num_update>1 时，cur_decay 会变大 -> 更强调历史（更新更慢、更平滑）
        """
        self.cur_decay = min(self.cur_decay * num_update, self.max_decay)

    def forward(self, inputs_flatten: torch.Tensor):
        """
        输入:
          inputs_flatten: (N, D)
            - N 通常是 B*H*W 或者 B*L
            - D = token_dim

        输出:
          quantize: (N, D) 量化后的特征（STE 让梯度回到 inputs_flatten）
          encoding_indices: (N, 1) 每个输入对应的 token id
        """

        # -----------------------------------------
        # 1) 计算 L2 距离到 codebook（K 个 token）
        # distances: (N, K)
        # 公式: ||x||^2 + ||e||^2 - 2 x·e
        # -----------------------------------------
        distances = (
            torch.sum(inputs_flatten**2, dim=1, keepdim=True)       # (N,1)
            + torch.sum(self.embed.data**2, dim=1)                  # (K,)
            - 2 * torch.matmul(inputs_flatten, self.embed.data.t()) # (N,K)
        )

        # -----------------------------------------
        # 2) 最近邻分配：对每个 x 选距离最小的 token
        # encoding_indices: (N,1)
        # -----------------------------------------
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # -----------------------------------------
        # 3) 生成 one-hot 编码 encodings: (N, K)
        # encodings[n, k] = 1 表示第 n 个输入分配到第 k 个 token
        # -----------------------------------------
        encodings = torch.zeros(
            encoding_indices.shape[0],  # N
            self.num_tokens,            # K
            dtype=torch.float,
            device=inputs_flatten.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # =========================================================
        # 训练模式：更新码本（EMA + 分布式同步）
        # =========================================================
        if self.training:
            # -----------------------------------------------------
            # 3.1) 统计每个 token 被选中的次数（本卡）
            # tmp_sum: (1, K)
            # -----------------------------------------------------
            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)

            # 分布式同步：收集所有卡的 tmp_sum，然后 sum 得到全局计数
            # concat_all_gather(tmp_sum) -> (world_size, K)
            # encoding_sum: (K,)
            encoding_sum = torch.sum(concat_all_gather(tmp_sum), dim=0)

            # cluster_sum 做“累计计数”（更像 total count）
            sum_inplace(self.cluster_sum, encoding_sum)

            # cluster_size 做 EMA 计数（平滑的 count）
            ema_tensor_inplace(self.cluster_size, encoding_sum, self.cur_decay)

            # -----------------------------------------------------
            # 3.2) 统计每个 token 分配到的特征和（本卡）
            # embed_sum_tmp: (K, D) = encodings^T (K,N) @ inputs_flatten (N,D)
            # -----------------------------------------------------
            embed_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)

            # 分布式同步所有卡的 embed_sum_tmp：
            # 这里先 unsqueeze(0) 变成 (1,K,D) 以便 concat_all_gather 拼 dim=0
            # concat -> (world_size, K, D) 再 sum(dim=0) -> (K,D)
            embed_sum = torch.sum(concat_all_gather(embed_sum_tmp.unsqueeze(dim=0)), dim=0)

            # embed_avg EMA 更新（平滑的 sum）
            ema_tensor_inplace(self.embed_avg, embed_sum, self.cur_decay)

            # -----------------------------------------------------
            # 3.3) 计算“平滑后的 cluster_size”，避免 0 导致除 0
            #
            # laplace_smoothing(self.cluster_size, K) -> 一个分布 p(k)，sum=1
            # 再乘以 self.cluster_size.sum()，把它缩放回“总计数尺度”
            #
            # cluster_size: (K,)
            # -----------------------------------------------------
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.num_tokens, self.eps)
                * self.cluster_size.sum()
            )

            # -----------------------------------------------------
            # 3.4) 用 EMA 的平均和 / 平滑后的计数 得到新的码本向量
            # embed_normalized: (K, D)
            # -----------------------------------------------------
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

            # -----------------------------------------------------
            # 3.5) 分布式同步：让所有 rank 的 embed 结果一致
            # 这里是 all_reduce(mean)：
            #   embed_normalized /= world_size
            #   all_reduce(sum)
            # 等价于全局平均
            # -----------------------------------------------------
            world_size = dist.get_world_size()
            dist.all_reduce(embed_normalized.div_(world_size))

            # 把新码本写回 self.embed（buffer）
            self.embed.data.copy_(embed_normalized)

        # =========================================================
        # 4) 量化：用 one-hot 从 embed 里选出对应 token 向量
        # quantize: (N, D) = encodings (N,K) @ embed (K,D)
        # =========================================================
        quantize = torch.matmul(encodings, self.embed)

        # =========================================================
        # 5) STE（straight-through estimator）
        #
        # 前向：quantize 的数值生效（离散选择）
        # 反向：梯度从 quantize 直接“穿透”回 inputs_flatten
        #
        # (quantize - inputs).detach(): 不让这部分传梯度
        # + inputs_flatten: 梯度等价于对 inputs_flatten 求导
        # =========================================================
        quantize = (quantize - inputs_flatten).detach() + inputs_flatten

        return quantize, encoding_indices
