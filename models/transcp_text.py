import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor


class LSTMBert(nn.Module):
    """
    文本编码器：BERT(多层输出) + GRU(序列建模) + Phrase Attention(对 token 加权汇聚得到 phrase/句子向量)

    典型用途：视觉-语言任务中，把输入表达(referring expression)编码为：
      1) token-level 特征 xs: (B, L, 768)
      2) phrase-level 全局特征 y: (B, 1, 768)  (注意：这里 y 的通道来自 embedded 的 768，不是 GRU hidden)
    """
    def __init__(self,
                 lstm_cfg=dict(type='gru',
                               num_layers=1,
                               dropout=0.,
                               hidden_size=512,
                               bias=True,
                               bidirectional=True,
                               batch_first=True),
                 output_cfg=dict(type="max"),
                 train_bert=True,
                 enc_num=12,
                 bert_model=None):
        super(LSTMBert, self).__init__()
        self.enc_num = enc_num              # 取 BERT 第 enc_num 层的输出（1-based 使用时注意）
        self.fp16_enabled = False           # 这里可能是兼容 mmcv 的标记，但本段代码没用到

        # -------------------------
        # 1) GRU/LSTM 配置
        # -------------------------
        # lstm_cfg 里必须指定 type='gru'（原作者只支持 gru）
        assert lstm_cfg.pop('type') in ['gru']
        # GRU 的 input_size 固定为 768（BERT-base 输出维度）
        # 注意：nn.GRU 的 input_size 是显式参数，这里写在后面覆盖传入
        self.lstm = nn.GRU(**lstm_cfg, input_size=768)

        # -------------------------
        # 2) 输出聚合方式配置（本实现里 output_type 实际没用上）
        # -------------------------
        output_type = output_cfg.pop('type')
        assert output_type in ['mean', 'default', 'max']
        self.output_type = output_type

        # -------------------------
        # 3) BERT 模型（外部传入）
        # -------------------------
        self.bert = bert_model

        # 是否训练 BERT（由 lr_bert > 0 决定）
        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

        # BERT token embedding 通道数（bert-base = 768）
        self.num_channels = 768

        # GRU 是否双向：双向时 hidden state 会有两个方向拼接
        # num_dirs=2 表示 bidirectional=True，否则 1
        if lstm_cfg["bidirectional"]:
            self.num_dirs = 2
        else:
            self.num_dirs = 1

        # Phrase attention：对 “GRU 输出(context)” 计算每个 token 的注意力权重，
        # 再用权重对 embedded(xs) 做加权求和，得到句子/短语表示。
        #
        # 注意：这里 attention 的打分输入维度是 GRU 输出维度：
        #   GRU 输出维度 = hidden_size * num_dirs
        self.sub_attn = PhraseAttention_Liu(self.lstm.hidden_size * self.num_dirs)

    def forward(self, ref_expr_inds: NestedTensor):
        """
        Args:
            ref_expr_inds: NestedTensor
                - ref_expr_inds.tensors: LongTensor (B, L)
                  token ids，padding 的位置一般为 0
                - ref_expr_inds.mask:    BoolTensor (B, L)
                  注意：这里具体语义取决于你们的 NestedTensor 实现：
                  常见两种：
                    A) True=padding(要忽略)   (DETR 体系常用)
                    B) True=valid(要保留)
                  但你后面又自己算了 y_mask = (token==0)，所以最终以 y_mask 为准。

        Returns:
            y: Tensor (B, 1, 768)
               phrase-level 表示（对 embedded=xs 的加权和）
            NestedTensor(xs, y_mask):
               - xs:     Tensor (B, L, 768)  token-level BERT 表示（选定层）
               - y_mask: BoolTensor (B, L)   True 表示 padding（忽略）
        """

        # -------------------------
        # 1) BERT 编码
        # -------------------------
        # 这里假设 self.bert 的 forward 支持输出 “每层的 sequence_output 列表/tuple”
        # 即 sequence_output[layer] = (B, L, 768)
        #
        # attention_mask 传 ref_expr_inds.mask：
        #   如果你的 mask 是 True=padding，那这里其实应该传 (~mask).long()
        #   但由于后面 y_mask 重新定义，这里是否正确主要影响 BERT 内部 attention。
        sequence_output, pooled_output = self.bert(
            ref_expr_inds.tensors,
            attention_mask=ref_expr_inds.mask
        )

        # 取指定层的 token 表示：xs (B, L, 768)
        # enc_num=12 表示取第 12 层；由于索引从 0 开始，所以是 enc_num-1
        xs = sequence_output[self.enc_num - 1]

        # -------------------------
        # 2) 构造 padding mask（True 表示 padding/忽略）
        # -------------------------
        # token==0 视为 padding
        y_mask = torch.abs(ref_expr_inds.tensors) == 0  # (B, L), bool

        # -------------------------
        # 3) GRU 序列建模
        # -------------------------
        # y_word: (B, L, hidden_size*num_dirs)
        # h:      (num_layers*num_dirs, B, hidden_size)
        y_word, h = self.lstm(xs)

        # -------------------------
        # 4) Phrase Attention：从 token 里选“更重要”的词进行加权汇聚
        # -------------------------
        # context = y_word (B, L, Hc) 用于计算 attention score
        # embedded = xs    (B, L, 768) 用于被加权求和（得到 768 维 phrase 表示）
        # mask = y_mask    (B, L) True=padding 需要忽略
        #
        # 返回：
        #   sub_attn        (B, L)   每个 token 的注意力权重
        #   sub_phrase_emb  (B, 1, 768) 加权后的句子/短语向量
        sub_attn, sub_phrase_emb = self.sub_attn(y_word, xs, y_mask)

        # 这里把 phrase embedding 作为全局输出 y
        y = sub_phrase_emb

        # 返回 y 以及 token-level 表示 xs + mask
        return (y, NestedTensor(xs, y_mask))


class PhraseAttention_Liu(nn.Module):
    """
    简单的 token attention：
      1) 用 fc(context) 得到每个 token 的打分 score
      2) softmax 得到注意力权重 attn
      3) mask 掉 padding，并重新归一化
      4) 用 attn 对 embedded 做加权求和

    注意：
      - context 的维度是 GRU 输出维度 (hidden_size*num_dirs)
      - embedded 的维度是 BERT token embedding 维度 (768)
      - mask=True 表示 padding（要忽略）
    """
    def __init__(self, input_dim):
        super().__init__()
        # 将每个 token 的 context 向量映射成一个标量分数
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, mask):
        """
        Args:
            context : Tensor (B, L, Hc)  通常是 GRU 输出 y_word
            embedded: Tensor (B, L, 768) 通常是 BERT 输出 xs
            mask    : BoolTensor (B, L)  True=padding，需要忽略

        Returns:
            attn        : Tensor (B, L)       token 权重（padding 位置应接近 0）
            weighted_emb: Tensor (B, 1, 768)  加权汇聚的 phrase 向量
        """

        # 1) token 打分：fc -> (B, L, 1) 然后 squeeze -> (B, L)
        cxt_scores = self.fc(context).squeeze(2)

        # 2) softmax 得到注意力（未 mask 前）
        attn = F.softmax(cxt_scores, dim=-1)  # (B, L)

        # 3) 将 padding 位置权重置 0
        # (~mask) True 表示有效 token，转 float 就是 1/0
        attn = attn * ((~mask).float())  # (B, L)

        # 4) 重新归一化：保证每行 attn 之和为 1
        # 注意潜在坑：如果一整行全是 padding，sum 会是 0，导致 NaN
        # 这里没有 eps，理论上需要保证至少有一个非 padding token
        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (B, L)

        # 5) batch 矩阵乘： (B,1,L) x (B,L,768) -> (B,1,768)
        attn3 = attn.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embedded)

        return attn, weighted_emb


def build_LSTMBert(args, bert_model):
    """
    构建 LSTMBert：
      - train_bert 由 args.lr_bert > 0 决定
      - enc_num 使用 args.bert_output_layers（取第几层 BERT 输出）
    """
    # 注意，该函数里除了本注释外，皆为原作者所加
    train_bert = args.lr_bert > 0
    lstm_bert = LSTMBert(
        enc_num=args.bert_output_layers,
        train_bert=train_bert,
        bert_model=bert_model
    )
    return lstm_bert
