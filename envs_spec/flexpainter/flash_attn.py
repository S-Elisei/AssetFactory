"""flash_attn.flash_attn_varlen_qkvpacked_func on torch's scaled_dot_product_attention. Requires all sequences in
cu_seqlens to have the same length and no dropout."""
import torch.nn.functional as F


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
    _, _, h, d = qkv.shape
    n = int(cu_seqlens[1] - cu_seqlens[0])
    q, k, v = qkv.reshape(-1, n, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    return F.scaled_dot_product_attention(q, k, v, scale=softmax_scale).transpose(1, 2).reshape(-1, h, d)
