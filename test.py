import torch
import torch.nn as nn

class HistCausalEncoder(nn.Module):
    def __init__(self, in_features, act_embed_dim, out_features, num_hist):
        super().__init__()
        self.in_features = in_features
        self.act_embed_dim = act_embed_dim
        self.out_features = out_features
        self.num_hist = num_hist

        concat_features = in_features + num_hist * (out_features + act_embed_dim)
        self.linear = nn.Linear(concat_features, out_features)

    def forward(self, o, act):
        """
        o:   (B, T, P, in_features)
        act: (B, T, P, act_embed_dim)
        return:
        z:   (B, T, P, out_features)
        """
        B, T, P, _ = o.shape
        H = self.num_hist
        device, dtype = o.device, o.dtype

        # 历史缓存 (B, P, H, out_features) 和 (B, P, H, act_embed_dim)
        z_hist = torch.zeros(B, P, H, self.out_features, device=device, dtype=dtype)
        a_hist = torch.zeros(B, P, H, self.act_embed_dim, device=device, dtype=dtype)

        z_list = []
        for t in range(T):
            # 展平历史 (B, P, H*(out_features + act_embed_dim))
            hist_flat = torch.cat([z_hist, a_hist], dim=-1).reshape(B, P, -1)

            # 当前输入 (B, P, concat_features)
            x = torch.cat([hist_flat, o[:, t]], dim=-1)

            # 直接调用 Linear: 会在最后一维上做映射 (patch 维度保持独立)
            z_t = self.linear(x)  # (B, P, out_features)
            z_list.append(z_t)

            # 更新历史
            if H > 0:
                if H > 1:
                    z_hist[:, :, :-1] = z_hist[:, :, 1:].clone()
                    a_hist[:, :, :-1] = a_hist[:, :, 1:].clone()
                z_hist[:, :, -1] = z_t
                a_hist[:, :, -1] = act[:, t]

        z = torch.stack(z_list, dim=1)  # (B, T, P, out_features)
        return z


# 使用示例
if __name__ == "__main__":
    B, T, P = 2, 5, 4
    in_features = 16
    act_embed_dim = 8
    out_features = 32
    num_hist = 3

    model = HistCausalEncoder(in_features, act_embed_dim, out_features, num_hist)
    o = torch.randn(B, T, P, in_features)
    act = torch.randn(B, T, P, act_embed_dim)
    z = model(o, act)
    print(z.shape)  # (2, 5, 4, 32)