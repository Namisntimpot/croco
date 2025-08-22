import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def clipped_softmax(data, dim=1, eta=1.1, gamma=-0.1, **kw):
    sm_out = torch.nn.functional.softmax(data, dim=dim, **kw)
    stretched_out = sm_out * (eta - gamma) + gamma
    return torch.clip(stretched_out, 0, 1)


SOFTMAX_MAPPING = {
    "vanilla": torch.nn.functional.softmax,
    # Clipped softmax
    "clipped(0:1.0003)": partial(clipped_softmax, gamma=0, eta=1.0003),
    "clipped(0:1.001)": partial(clipped_softmax, gamma=0, eta=1.001),
    "clipped(0:1.002)": partial(clipped_softmax, gamma=0, eta=1.002),
    "clipped(0:1.003)": partial(clipped_softmax, gamma=0, eta=1.003),
    "clipped(0:1.004)": partial(clipped_softmax, gamma=0, eta=1.004),
    "clipped(0:1.01)": partial(clipped_softmax, gamma=0, eta=1.01),
    "clipped(0:1.02)": partial(clipped_softmax, gamma=0, eta=1.02),
    "clipped(0:1.03)": partial(clipped_softmax, gamma=0, eta=1.03),
    "clipped(0:1.1)": partial(clipped_softmax, gamma=0, eta=1.1),
    "clipped(-.1:1)": partial(clipped_softmax, gamma=-0.1, eta=1.0),
    "clipped(-.00001:1)": partial(clipped_softmax, gamma=-0.00001, eta=1.0),
    "clipped(-.00003:1)": partial(clipped_softmax, gamma=-0.00003, eta=1.0),
    "clipped(-.0001:1)": partial(clipped_softmax, gamma=-0.0001, eta=1.0),
    "clipped(-.0003:1)": partial(clipped_softmax, gamma=-0.0003, eta=1.0),
    "clipped(-.0005:1)": partial(clipped_softmax, gamma=-0.0005, eta=1.0),
    "clipped(-.001:1)": partial(clipped_softmax, gamma=-0.001, eta=1.0),
    "clipped(-.002:1)": partial(clipped_softmax, gamma=-0.002, eta=1.0),
    "clipped(-.0025:1)": partial(clipped_softmax, gamma=-0.0025, eta=1.0),
    "clipped(-.003:1)": partial(clipped_softmax, gamma=-0.003, eta=1.0),
    "clipped(-.004:1)": partial(clipped_softmax, gamma=-0.004, eta=1.0),
    "clipped(-.005:1)": partial(clipped_softmax, gamma=-0.005, eta=1.0),
    "clipped(-.01:1)": partial(clipped_softmax, gamma=-0.01, eta=1.0),
    "clipped(-.015:1)": partial(clipped_softmax, gamma=-0.015, eta=1.0),
    "clipped(-.02:1)": partial(clipped_softmax, gamma=-0.02, eta=1.0),
    "clipped(-.025:1)": partial(clipped_softmax, gamma=-0.025, eta=1.0),
    "clipped(-.03:1)": partial(clipped_softmax, gamma=-0.03, eta=1.0),
    "clipped(-.04:1)": partial(clipped_softmax, gamma=-0.04, eta=1.0),
    "clipped(-.001:1.001)": partial(clipped_softmax, gamma=-0.001, eta=1.001),
    "clipped(-.002:1.002)": partial(clipped_softmax, gamma=-0.002, eta=1.002),
    "clipped(-.003:1.003)": partial(clipped_softmax, gamma=-0.003, eta=1.003),
    "clipped(-.005:1.005)": partial(clipped_softmax, gamma=-0.003, eta=1.005),
    "clipped(-.01:1.01)": partial(clipped_softmax, gamma=-0.01, eta=1.01),
    "clipped(-.03:1.03)": partial(clipped_softmax, gamma=-0.03, eta=1.03),
    "clipped(-.1:1.1)": partial(clipped_softmax, gamma=-0.1, eta=1.1),
}


class PerHeadLinear(nn.Module):
    def __init__(self, n_heads, in_features, out_features, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((n_heads, in_features, out_features))
        )
        self.apply_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.empty((n_heads, out_features))
            )
        else:
            self.register_parameter("bias", None)
        self.initialize_weight()

    def initialize_weight(self):
        nn.init.xavier_uniform_(self.weight)
        if self.apply_bias:
            nn.init.constant_(self.bias, 0.)
        
    def forward(self, x):
        '''
        x: b, n, h, d  
        return: b, n, h, o_feat
        '''
        x = torch.einsum("bnhd,hdo -> bnho", x, self.weight)
        if self.apply_bias:
            x = x + self.bias
        return x


class PerHeadMlp(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=0.25, use_linear=False):
        '''
        PerHeadLinear implementation is about 8.5x faster than common linear implementation
        '''
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.mlp_ratio = mlp_ratio
        self.use_linear = use_linear
        if use_linear:
            module_list = []
            for _ in range(n_heads):
                if mlp_ratio > 0:
                    hid_dim = int(mlp_ratio * self.head_dim)
                    fc = nn.Sequential(
                        nn.Linear(self.head_dim, hid_dim, True), nn.ReLU(), nn.Linear(hid_dim, 1, True)
                    )
                else:
                    fc = nn.Linear(self.head_dim, 1, True)
                module_list.append(fc)
            self.mlp = nn.ModuleList(module_list)
        else:
            if mlp_ratio > 0:
                hid_dim = int(mlp_ratio * self.head_dim)
                self.mlp = nn.Sequential(
                    PerHeadLinear(n_heads, self.head_dim, hid_dim, True),
                    nn.ReLU(),
                    PerHeadLinear(n_heads, hid_dim, 1, True)
                )
            else:
                self.mlp = PerHeadLinear(n_heads, self.head_dim, 1, True)

        self.forward_func = self.forward_common_linear if use_linear else self.forward_per_head_linear

    def forward_common_linear(self, x):
        '''x: b, n, h, d
        out: b, n, h, 1
        '''
        alpha_head = []
        for i in range(self.n_heads):
            head_feat = x[:,:,i]  # b, n, d
            mlp = self.mlp[i]
            head_gate = mlp(head_feat)  # b, n, 1
            alpha_head.append(head_gate)
        return torch.stack(alpha_head, dim=-2)  # b, n, h, 1
    
    def forward_per_head_linear(self, x):
        '''x: b, n, h, d
        out: b, n, h, 1
        '''
        return self.mlp(x)

    def forward(self, x):
        '''x: b, n, h, d
        out: b, n, h, 1
        '''
        return self.forward_func(x)



class VGA(nn.Module):
    def __init__(
            self,
            dim:int,
            num_heads:int,
            gate_type:str = 'cond_per_head',  # ['uncond', 'cond_per_head', 'cond_per_token', 'cond_all']
            mlp_ratio:float = 0.25,   # 0 for only 1 linear layer, without hidden layer.
            # gate_init:float = None,
            gate_fn = F.sigmoid,
            gate_by_all_feat:bool = False,
        ):
        super().__init__()
        self.dim = dim
        self.n_heads = num_heads
        self.head_dim = dim // num_heads

        assert gate_type in ['uncond', 'cond_per_head', 'cond_per_token']
        self.gate_type = gate_type
        self.gate_by_all_feat = gate_by_all_feat
        if self.gate_type == 'uncond':
            self.alpha = nn.Parameter(torch.zeros(self.n_heads))
            torch.nn.init.normal_(self.alpha, std=0.02)
        # elif self.gate_type == 'cond_all':
        #     self.alpha = nn.Linear(
        #         self.dim, self.n_heads, bias=True
        #     )
        elif self.gate_type == 'cond_per_token' or self.gate_type == 'cond_per_head':
            if self.gate_by_all_feat:
                self.alpha = nn.Linear(self.dim, self.n_heads, bias=True)
            else:
                self.alpha = PerHeadMlp(self.dim, self.n_heads, mlp_ratio, False)  # which is faster?
        
        self.gate_fn = gate_fn
        
    def forward(self, x:torch.Tensor, attn_out:torch.Tensor):
        '''
        x & attn_out: (b, n, h, d)  
        if this module is supposed to works as a "value-gated attention" mechanism, x should be the V tensor in the attention operation.
        otherwise, x should be the original feature (before qkv projection).
        '''
        b, n, h, d = x.shape
        if self.gate_type == 'uncond':
            gate = self.gate_fn(self.alpha).view(1, -1, 1) # (1,h,1)
        else:
            if self.gate_by_all_feat:
                x = x.view(b, n, -1)
                gate = self.alpha(x).unsqueeze(-1)  # b, n, h, 1
            else:
                gate = self.alpha(x)                # b, n, h, 1
            if self.gate_type == 'cond_per_head':
                gate = torch.mean(gate, dim=1, keepdim=True)  # b, 1, h, 1
            gate = self.gate_fn(gate)  # b, n(or 1), h, 1
        
        attn_out = attn_out * gate
        return attn_out


if __name__ == '__main__':
    from time import time
    from tqdm import tqdm
    b = 256
    n = 196
    h = 12
    total_d = 768
    d = total_d // h
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    common_mlp = PerHeadMlp(total_d, h, 0.25, True).to(device)
    perhead_mlp = PerHeadMlp(total_d, h, 0.25, False).to(device)
    # perhead_mlp.apply(lambda m: print(type(m)))

    dummy = torch.rand((b,n,h,d), dtype=torch.float32, device=device)
    
    warmup = 50
    test = 50
    t = 0
    cnt = 0
    for i in tqdm(range(warmup + test), desc="common mlp"):
        s = time()
        x = common_mlp.forward(dummy)
        interval = time() - s
        if i >= warmup:
            t += interval
            cnt += 1
    print("common mlp: ", t / cnt)
    t = 0
    cnt = 0
    for i in tqdm(range(warmup + test), desc="perhead mlp"):
        s = time()
        x = perhead_mlp.forward(dummy)
        interval = time() - s
        if i >= warmup:
            t += interval
            cnt += 1
    print("perhead mlp: ", t / cnt)