import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.registry import ATTENTION


@ATTENTION.register_module()
class GeometryDecoupledAttention(BaseModule):
    """
    Implementation of Geometry-Decoupled Attention (GDA).
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_ins (int): The number of instances.
        num_pts (int): The number of fixed points of each instance.
            -----------------------------------------------
            The indices of queries is organized as follows:

            Instance 0: {0, 1, ..., num_pts-1}
            Instance 1: {num_pts, num_pts+1, ..., 2*num_pts-1}
            ...
            Instance num_ins-1: {(num_ins-1)*num_pts, (num_ins-1)*num_pts+1, ..., num_ins*num_pts-1}
            -----------------------------------------------
        embed_dims: embedding dimension
        num_heads: the number of attention heads
        dropout: dropout rate used in attention
    """

    def __init__(
            self,
            num_ins=50,
            num_pts=20,
            embed_dims=256,
            num_heads=8,
            dropout=0.1,
            **kwargs,
        ):
        super(GeometryDecoupledAttention, self).__init__()
        self.num_ins = num_ins
        self.num_pts = num_pts
        self.embed_dims = embed_dims
        self.intra_attention = nn.MultiheadAttention(embed_dim=embed_dims, num_heads=num_heads, dropout=dropout)
        self.intra_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims)
        )
        self.intra_norm1 = nn.LayerNorm(embed_dims)
        self.intra_norm2 = nn.LayerNorm(embed_dims)
        self.inter_attention = nn.MultiheadAttention(embed_dim=embed_dims, num_heads=num_heads, dropout=dropout)
        self.inter_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims)
        )
        self.inter_norm1 = nn.LayerNorm(embed_dims)
        self.inter_norm2 = nn.LayerNorm(embed_dims)
    
    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        **kwargs
    ):
        """
            The input query is of shape (num_ins * num_pts, B, embed_dims)
        """
        device = query.device
        intra_mask = torch.ones((self.num_ins * self.num_pts, self.num_ins * self.num_pts), device=device, dtype=torch.bool)
        inter_mask = torch.zeros((self.num_ins * self.num_pts, self.num_ins * self.num_pts), device=device, dtype=torch.bool)
        for i in range(self.num_ins):
            intra_mask[i*self.num_pts:(i+1)*self.num_pts, i*self.num_pts:(i+1)*self.num_pts] = False
            inter_mask[i*self.num_pts:(i+1)*self.num_pts, i*self.num_pts:(i+1)*self.num_pts] = True

        out_inter, _ = self.inter_attention(
            query = query,
            key = query,
            value = query,
            attn_mask = inter_mask,
        )
        out_inter = self.inter_norm1(out_inter) + query
        out_inter = self.inter_norm2(self.inter_ffn(out_inter)) + out_inter
        out_intra, _ = self.intra_attention(
            query = out_inter,
            key = out_inter,
            value = out_inter,
            attn_mask = intra_mask,
        )
        out_intra = self.intra_norm1(out_intra) + out_inter
        out_intra = self.intra_norm2(self.intra_ffn(out_intra)) + out_intra
        return out_intra