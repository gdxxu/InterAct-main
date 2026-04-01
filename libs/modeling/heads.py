import torch
from torch import nn
from torch.nn import functional as F

from .blocks import LayerNorm, get_sinusoid_encoding, MaskedConv1D

import copy
import math

#####################################################################################################
# STI
def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel)
        self.dropout = nn.Dropout(drop_out)

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1
        
    def forward(self, queries, keys, values, mask):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
        values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values)  
        Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6)

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) 
 
        x = x.reshape(B, L, -1) 
        x = self.out_projection(x)
        x = self.dropout(x)

        return x * mask[:, 0, :, None]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            )
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x)
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
       
        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1) 
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features))
        
    def forward(self, feature_s, feature_t, mask):
        feature_s = feature_s.permute(0, 2, 1)
        n, c, t = feature_s.shape
        feature_s = self.conv_s(feature_s)
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t)/t)
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t)
        feature_cross = feature_cross + feature_t
        feature_cross = feature_cross.permute(0, 2, 1)
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask
    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.3, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer
        num_SFI_layers = len(SFI_layer)
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)

        self.conv_in = nn.Conv2d(in_channel, num_SFI_layers+1, kernel_size=1)
        self.conv_t = nn.Conv1d(node, n_features, 1)
        self.SFI_layers = nn.ModuleList(
            [SFI(node, n_features) for i in range(num_SFI_layers)])
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in 
                range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        count = 0
        x = self.conv_in(x)
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers), 1), dim=1)
        feature_t = feature_t.squeeze(1).permute(0, 2, 1)
        feature_st = self.conv_t(feature_t)

        for index, layer in enumerate(self.layers):
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count,:], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, None, mask)

        feature_st = self.conv_out(feature_st)
        return feature_st * mask

class Decoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        out_channel: int,
        num_layers: int,
        alpha = 1,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList([
            AttModule(2**i, n_features, n_features, 'decoder', alpha) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)
    
    def forward(self, x, fencoder, mask):
        """
        input:
            x: [B, n_classes, T]
            fencoder: [B, n_features, T]
            mask: [B, 1, T]
        output:
            out_cls: [B, n_classes, T]
            out_feat: [B, n_features, T]
        """
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        return out
        # return out, feature
    
class DecoderPyramid(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        out_channel: int,
        num_layers: int,
        alpha = 1,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList([
            AttModule(2**i, n_features, n_features, 'decoder', alpha) for i in range(num_layers)
        ])
        self.act = nn.PReLU(n_features * num_layers)
        self.conv1_1 = nn.Conv1d(n_features*num_layers, n_features, 1)
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)
    
    def forward(self, x, fencoder, mask):
        """
        input:
            x: [B, n_classes, T]
            fencoder: [B, n_features, T]
            mask: [B, 1, T]
        output:
            out_cls: [B, n_classes, T]
            out_feat: [B, n_features, T]
        """
        out = []
        in_feat = self.conv_in(x)
        for layer in self.layers:
            if len(out) == 0:
                out.append(layer(in_feat, fencoder, mask))
            else:
                out.append(layer(in_feat + out[-1], fencoder, mask))
        out_cat = self.act(torch.cat(out, dim=1))
        out_feat = self.conv1_1(out_cat)
        out_cls = self.conv_out(out_feat + in_feat)

        return out_cls * mask[:, 0:1, :]

class DialtedResidualLayer(nn.Module):
    def __init__(
        self,
        dilation: int,
        in_channel: int,
        out_channel: int,
    ) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channel, out_channel, 3, stride=1, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
    
    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)

        return (x + out) * mask[:, 0:1, :]

class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DialtedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)
    
    def forward(self, x, mask):
        """
        input:
            x: [B, 1, T]
            mask: [B, 1, T]
        output:
            out_bound: [B, 1, T]
            out_feat: [B, n_features, T]
        """
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        x = self.conv_out(feature)

        return x * mask[:, 0:1, :]
    
class SingleStageETSPNet(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DialtedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.act = nn.PReLU(n_features * n_layers)
        self.conv1_1 = nn.Conv1d(n_features*n_layers, n_features, 1)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask):
        """
        input:
            x: [B, 1, T]
            mask: [B, 1, T]
        output:
            out_bound: [B, 1, T]
            out_feat: [B, n_features, T]
        """
        out = []
        in_feat = self.conv_in(x)
        for layer in self.layers:
            if len(out) == 0:
                out.append(layer(in_feat, mask))
            else:
                out.append(layer(in_feat + out[-1], mask))
        out_cat = self.act(torch.cat(out, dim=1))
        out_feat = self.conv1_1(out_cat)
        out_bound = self.conv_out(out_feat + in_feat)

        return out_bound * mask[:, 0:1, :]

class X2Y_map(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        y_out_dim: int,
        hid_dim: int,
        dropout: float = 0.0,
        with_query_pos: bool = False,
        with_frame_pos: bool = False,
    ) -> None:
        super().__init__()
        self.X_K = nn.Linear(x_dim, hid_dim)
        self.X_V = nn.Linear(x_dim, hid_dim)
        self.Y_Q = nn.Linear(y_dim, hid_dim)

        self.Y_W = nn.Linear(y_dim+hid_dim, y_out_dim)

        self.dropout = nn.Dropout(dropout)
        self.with_query_pos = with_query_pos
        self.with_frame_pos = with_frame_pos

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor + pos.to(tensor.device) if pos is not None else tensor     

    def forward(self, X_feature, Y_feature, X_pos=None, Y_pos=None):
        """
        input:
            X_feature: query feature [B, embd_dim, num_classes/1]
            Y_feature: frame feature [B, embd_dim, T]
            X_pos: query pos [B, embd_dim, num_classes/1] / None
            Y_pos: frame pos [1, embd_dim, T'] / None
        output:
            out: update frame feature [B, y_out_dim, T]
        """
        X_feature = self.with_pos_embed(X_feature, X_pos) if self.with_query_pos else X_feature
        Y_feature = self.with_pos_embed(Y_feature, Y_pos) if self.with_frame_pos else Y_feature

        X_feature = X_feature.transpose(1, 2)  # [B, num_classes/1, embd_dim]
        Y_feature = Y_feature.transpose(1, 2)  # [B, T, embd_dim]

        xk = self.X_K(X_feature)  # [B, num_classes/1, hid_dim]
        xv = self.X_V(X_feature)  # [B, num_classes/1, hid_dim]

        yq = self.Y_Q(Y_feature)  # [B, T, hid_dim]

        attn = torch.einsum('bqd, bkd -> bqk', yq, xk) / (xk.shape[-1] ** 0.5)  # [B, T, num_classes/1]
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bqk, bkd -> bqd', attn, xv)  # [B, T, hid_dim]
        out = self.Y_W(torch.cat([out, Y_feature], dim=-1))
        out = self.dropout(out)
        out_Y_feature = out.transpose(1, 2)  # [B, hid_dim, T]

        return out_Y_feature

#####################################################################################################

class ActionHead(nn.Module):
    def __init__(self,
                n_embd: int,            
                max_seq_len: int,       
                num_classes: int,
                n_head: int,            
                n_dec_layers: int,     
                n_blocks: int,         
                attn_pdrop=0.0,        # dropout rate for the attention map
                proj_pdrop=0.0,        # dropout rate for the projection / MLP
                path_pdrop=0.0,        # drop path rate
                with_feat_pos=False,    
                with_query_pos=False,   
                with_aux_outputs=False, 
                with_aux_boundary=False,
                normalize_before=False, 
                decoder_type='transformer',
                ):
        super().__init__()

        self.num_classes = num_classes
        self.n_dec_layers = n_dec_layers
        self.n_blocks = n_blocks
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.with_aux_outputs = with_aux_outputs
        self.with_aux_boundary = with_aux_boundary
        self.decocer_type = decoder_type

        self.feat_pe = []
        for i in range(self.n_blocks):
            pos_embd = get_sinusoid_encoding(int(self.max_seq_len/(2**i)), n_embd) / (n_embd**0.5)
            self.register_buffer(f"feat_pe_{i}", pos_embd, persistent=False)
            self.feat_pe.append(pos_embd)

        self.query = nn.Embedding(num_classes+1, n_embd)    
        self.query_pe = nn.Embedding(num_classes+1, n_embd)
        self.level_embd = nn.Embedding(n_blocks, n_embd)    
        
        if decoder_type == 'transformer':
            self.decoder_layer = TransformerdecoderLayer(
                    n_embd = n_embd,                    
                    n_head = n_head,                    
                    n_classes = num_classes,            
                    n_blocks = n_blocks,                
                    attn_pdrop = attn_pdrop,    
                    proj_pdrop = proj_pdrop,
                    path_pdrop = path_pdrop,
                    with_query_pos = with_query_pos,
                    with_feat_pos = with_feat_pos,
                    normalize_before = normalize_before,
                    )

        self.decoder_layers = nn.ModuleList([copy.deepcopy(self.decoder_layer) for _ in range(n_dec_layers)])
            
        self.mask_embd = nn.Sequential(     
                        nn.Conv1d(n_embd, n_embd//4, 1), nn.ReLU(inplace=True), 
                        nn.Conv1d(n_embd//4, n_embd, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(n_embd, n_embd, 1)
                        )
        self.boundary_embd = nn.Sequential( 
                        nn.Linear(n_embd, n_embd//4), nn.ReLU(inplace=True),
                        nn.Linear(n_embd//4, n_embd), nn.ReLU(inplace=True),
                        nn.Linear(n_embd, n_embd)
                        )

    def forward(self, mask_feat, feats, masks):
        masks = [x.unsqueeze(1) for x in masks] 
        lvl_feats = []
        ### add level pe
        for i in range(self.n_blocks): 
            lvl_feats.append(feats[i] + self.level_embd.weight[i][None, :, None])

        B = feats[0].size()[0]
        query = self.query.weight.transpose(0,1).unsqueeze(0).repeat(B,1,1)
        query_pe = self.query_pe.weight.transpose(0,1).unsqueeze(0).repeat(B,1,1)

        pred_mask_all = []
        pred_boundary_all = []
        mask_feature_all = []
        boundary_feature_all = []

        for decoder_leyer in self.decoder_layers:
            # pred_mask, pred_boundary = decoder_leyer\
            pred_mask, pred_boundary, mask_feature, boundary_feature = decoder_leyer\
                    (query, feats, masks, query_pe, self.feat_pe, mask_feat, self.forward_pred_mask, self.forward_pred_boundary)
            pred_mask_all += pred_mask
            pred_boundary_all += pred_boundary
            mask_feature_all += mask_feature
            boundary_feature_all += boundary_feature

        out_pred_mask = pred_mask_all if self.with_aux_outputs else pred_mask_all[-1]
        out_pred_boundary = pred_boundary_all if self.with_aux_boundary else pred_boundary_all[-1]   
        out_mask_feature = mask_feature_all if self.with_aux_outputs else mask_feature_all[-1]
        out_boundary_feature = boundary_feature_all if self.with_aux_boundary else boundary_feature_all     

        return out_pred_mask, out_pred_boundary
        # return out_pred_mask, out_pred_boundary, out_mask_feature, out_boundary_feature


    def forward_pred_mask(self, action_query, mask_feat):    
        mask_embd = self.mask_embd(action_query)
        output_mask = torch.einsum('bcn, bcl -> bnl', mask_embd, mask_feat)

        return output_mask
    

    def forward_pred_boundary(self, boundary_query, mask_feat):
        boundary_embd = self.boundary_embd(boundary_query)
        boundary = torch.einsum('bc, bcl -> bl', boundary_embd, mask_feat)

        return boundary


class TransformerdecoderLayer(nn.Module):
    def __init__(self,
            n_embd,                # dimension of the input features
            n_head,                # number of attention heads
            n_classes,             # number of classes
            n_blocks,              
            n_hidden=None,         # dimension of the hidden layer in MLP
            activation="relu",     # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,        # dropout rate for the attention map
            proj_pdrop=0.0,        # dropout rate for the projection / MLP
            path_pdrop=0.0,        # drop path rate
            with_query_pos=False,   
            with_feat_pos=False,    
            normalize_before=False, 
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_blocks = n_blocks
        self.with_query_pos = with_query_pos
        self.with_feat_pos = with_feat_pos
        n_hidden = n_embd * 4 if n_hidden is None else n_hidden

        self.asb = Decoder(n_embd, n_embd, n_embd, num_layers=10, alpha=1)
        self.brb = SingleStageETSPNet(n_embd, n_embd, n_embd, n_layers=10)

        self.cross_attn = CrossAttentionLayer(n_embd, n_head, attn_pdrop, path_pdrop, activation, with_query_pos, with_feat_pos, normalize_before)
        self.self_attn = SelfAttentionLayer(n_embd, n_head, attn_pdrop, path_pdrop, activation, with_query_pos, normalize_before)
        self.ffn = FFNLayer(n_embd, n_hidden, proj_pdrop, activation, normalize_before)

        self.cross_attn_layers = nn.ModuleList([copy.deepcopy(self.cross_attn) for _ in range(n_blocks)])
        self.self_attn_layers = nn.ModuleList([copy.deepcopy(self.self_attn) for _ in range(n_blocks)])
        self.ffn_layers = nn.ModuleList([copy.deepcopy(self.ffn) for _ in range(n_blocks)])

        self.q2mask = X2Y_map(n_embd, n_embd, n_embd, n_embd, attn_pdrop, with_query_pos, with_feat_pos)
        self.q2boundary = X2Y_map(n_embd, n_embd, n_embd, n_embd, attn_pdrop, with_query_pos, with_feat_pos)
        self.q2mask_layers = nn.ModuleList([copy.deepcopy(self.q2mask) for _ in range(n_blocks)])
        self.q2boundary_layers = nn.ModuleList([copy.deepcopy(self.q2boundary) for _ in range(n_blocks)])

    def forward(self, query, feats, masks, query_pe, feat_pe, mask_feat, mask_func, boundary_func):
        pred_mask = []
        pred_boundary = []
        mask_feature_all = []
        boundary_feature_all = []

        output_mask = mask_func(query[...,:-1], mask_feat)  
        pred_mask.append(output_mask)

        output_boundary = boundary_func(query[...,-1], mask_feat)   
        pred_boundary.append(output_boundary)

        mask_feature = self.asb(mask_feat, mask_feat, masks[0])
        boundary_feature = self.brb(mask_feat, masks[0])
        mask_feature_all.append(mask_feature)
        boundary_feature_all.append(boundary_feature)

        for i in range(self.n_blocks-1, -1, -1):

            query = self.self_attn_layers[i](query, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pe)   
            query = self.cross_attn_layers[i](query, feats[i], None, ~masks[i].squeeze(1), feat_pe[i], query_pe)

            mask_feature = self.q2mask_layers[i](query, mask_feature, query_pe, feat_pe[0])
            boundary_feature = self.q2boundary_layers[i](query, boundary_feature, query_pe, feat_pe[0])

            mask_feature_all.append(mask_feature)
            boundary_feature_all.append(boundary_feature)

            output_mask = mask_func(query[...,:-1], mask_feature)      # mask_feature
            pred_mask.append(output_mask)

            output_boundary = boundary_func(query[...,-1], boundary_feature)   # boundary_feature
            pred_boundary.append(output_boundary)

        # return pred_mask, pred_boundary
        return pred_mask, pred_boundary, mask_feature_all, boundary_feature_all


class CrossAttentionLayer(nn.Module):
    def __init__(self, 
            d_model,                
            nhead,                  
            attn_pdrop=0.0, 
            path_pdrop=0.0, 
            activation="relu", 
            with_query_pos=False,   
            with_feat_pos=False,    
            normalize_before=False  
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_pdrop, batch_first=True)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(path_pdrop)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.with_query_pos = with_query_pos
        self.with_feat_pos = with_feat_pos

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor + pos.to(tensor.device) if pos is not None else tensor
    
    def forward_post(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        query = self.with_pos_embed(tgt, query_pos) if self.with_query_pos else tgt
        key = self.with_pos_embed(memory, pos) if self.with_feat_pos else memory
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        memory = memory.transpose(1,2)
        tgt2 = self.multihead_attn(query=query,
                                    key=key,
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.transpose(1,2)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos= None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        query = self.with_pos_embed(tgt2, query_pos) if self.with_query_pos else tgt2
        key = self.with_pos_embed(memory, pos) if self.with_feat_pos else memory
        query = query.transpose(1,2)    # [B, n_embd, num_classes+1] -> [B, num_classes+1, n_embd]
        key = key.transpose(1,2)        # [B, n_embd, T'] -> [B, T', n_embd]
        memory = memory.transpose(1,2)  # [B, n_embd, T'] -> [B, T', n_embd]
        tgt2 = self.multihead_attn(query=query,
                                    key=key,
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.transpose(1,2)      # [B, num_classes+1, n_embd] -> [B, n_embd, num_classes+1]
        tgt = tgt + self.dropout(tgt2)  # [B, n_embd, num_classes+1]

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        """
        tgt: B, n_embd, num_classes+1
        memory: [B, n_embd, T']
        memory_mask: None
        memory_key_padding_mask: ~masks[i].squeeze(1)
        pos: [1, n_embd, T']
        query_pos: B, n_embd, num_classes+1
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class SelfAttentionLayer(nn.Module):
    def __init__(self, 
            d_model,                
            nhead,                  
            attn_pdrop=0.0,     
            path_pdrop=0.0,
            activation="relu",  
            with_query_pos=False,   
            normalize_before=False  
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_pdrop, batch_first=True)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(path_pdrop)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.with_query_pos = with_query_pos

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos=None):
        pos_len = pos.shape[-1]
        if pos is None:
            return tensor
        if pos_len == tensor.shape[-1]:
            return tensor + pos.to(tensor.device)
        else:
            tensor[:,:,:pos_len] += pos
            return tensor

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos) if self.with_query_pos else tgt
        q = k = q.transpose(1,2)
        tgt2 = self.self_attn(query=q, key=k, value=tgt.transpose(1,2), attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2.transpose(1,2))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) if self.with_query_pos else tgt2
        q = k = q.transpose(1,2)
        tgt2 = self.self_attn(q, k, value=tgt2.transpose(1,2), attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2.transpose(1,2))
        
        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        """
        tgt: B, n_embd, num_classes+1
        query_pos: B, n_embd, num_classes+1
        """
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, 
            d_model,                
            dim_feedforward=2048, 
            path_pdrop=0.0,
            activation="relu", 
            normalize_before=False  
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(path_pdrop)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt = tgt.transpose(1,2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = tgt.transpose(1,2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = tgt2.transpose(1,2)  # [B, n_embd, num_classes+1] -> [B, num_classes+1, n_embd]
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2.transpose(1,2))   # [B, n_embd, num_classes+1]
        return tgt
    
    def forward(self, tgt):
        # tgt: B, n_embd, num_classes+1
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
