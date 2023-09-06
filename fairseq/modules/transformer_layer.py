# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from typing import Dict, List, Optional

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, MultimodelMultiheadAttention
from torch import Tensor

from fairseq.modules import MultiheadAttention_Image

import math


class HighWayNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.attention_dropout
        # self.lin_lay = nn.ModuleList([
        #     nn.Linear(params.hidden_size * 2, params.hidden_size * 2)
        #     for _ in range(2)])
        # self.gat_lay = nn.ModuleList([
        #     nn.Linear(params.hidden_size * 2, params.hidden_size * 2)
        #     for _ in range(2)])
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim * 2),
                                  nn.Sigmoid()))
        self.highway_linear = nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim)




    def forward(self, x, x1):

        x = torch.cat([x, x1], dim=-1)


        for i in range(2):
            h = getattr(self, 'highway_linear{}'.format(i))(x)
            g = getattr(self, 'highway_gate{}'.format(i))(x)
            x = g * h + (1 - g) * x
        x = self.highway_linear(x)
        x = nn.functional.dropout(x, self.dropout, self.training)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        # self.self_attn_img = MultiheadAttention(
        #     self.embed_dim,
        #     args.encoder_attention_heads,
        #     dropout=args.attention_dropout,
        #     self_attention=True,
        # )

        self.self_attn_txt_img = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        # self.fc1_image = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        # self.fc2_image = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        #        self.fc3 = Linear(args.embed_dim*2, args.embed_dim)
        self.FC = Linear(self.embed_dim, self.embed_dim)

        self.img_fc3 = Linear(self.embed_dim, self.embed_dim)
        self.img_fc4 = Linear(self.embed_dim, self.embed_dim)
        self.img_fc5 = Linear(self.embed_dim, self.embed_dim)



        # self.fc_con = Linear(2*self.embed_dim, self.embed_dim)
        self.fc_con_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.gating = GatingMechanism(args)
        # self.highway_net = HighWayNet(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def image_encoder(self, src_img_features,idx):

        # residual = src_img_features
        # src_img_features, _ = self.self_attn_img(
        #     query=src_img_features,
        #     key=src_img_features,
        #     value=src_img_features,
        #     key_padding_mask=None,
        #     attn_mask=None,
        # )

        residual = src_img_features

        if self.normalize_before:
            src_img_features = self.final_layer_norm(src_img_features)
        src_img_features = self.activation_fn(self.fc1_image(src_img_features))
        src_img_features = F.dropout(src_img_features, p=float(self.activation_dropout), training=self.training)
        src_img_features = self.fc2_image(src_img_features)
        src_img_features = F.dropout(src_img_features, p=self.dropout, training=self.training)
        src_img_features = residual + src_img_features
        if not self.normalize_before:
            src_img_features = self.final_layer_norm(src_img_features)

        return src_img_features
        #
    def text_image(self, text, src_img_features, encoder_padding_mask, batch_len):

        # text_semantic_att, _ = self.self_attn_txt_img(
        #     query=text[:batch_len],
        #     key=src_img_features,
        #     value=src_img_features,
        #     key_padding_mask=None,
        #     attn_mask=None,
        # )

        text_semantic_att, _ = self.self_attn_txt_img(
            query=src_img_features,
            key=text[:batch_len],
            value=text[:batch_len],
            key_padding_mask=None,
            attn_mask=None,
        )

        # text_semantic = text + text_semantic_att
        # if not self.normalize_before:
        #     text_semantic = self.self_attn_layer_norm(text_semantic)

        # residual = text_semantic
        # text_semantic = self.activation_fn(self.fc1_semantic(text_semantic))
        # text_semantic = F.dropout(text_semantic, p=float(self.activation_dropout), training=self.training)
        # text_semantic = self.fc2_semantic(text_semantic)
        # text_semantic = F.dropout(text_semantic, p=self.dropout, training=self.training)
        # text_semantic = residual + text_semantic
        # if not self.normalize_before:
        #     text_semantic = self.final_layer_norm(text_semantic)

        text_semantic_att = self.FC(text_semantic_att)

        return text_semantic_att

    # def data_normal(self,data):
    #     data_min = data.min()
    #     if data_min<0:
    #         data += torch.abs(data_min)
    #         data_min = data.min()
    #     data_max = data.max()
    #     dst = data_max - data_min
    #     normal_Data = (data-data_min).true_divide(dst)
    #     return normal_Data

    # def multimodel_mix(self, x, img, batch_len):
    #     pseudo_features = img[torch.Lon·gTensor(np.random.randint(0, img.size(0), batch_len))]
    #     x = self.data_normal(x) + 0.2 * self.data_normal(pseudo_features)
    #     return x
    def multimodel_mix(self, x, img, batch_len,layer_idx):
        #####  stra visual to textual  #########
        pseudo_features = img[torch.LongTensor(np.random.randint(0,img.size(0),batch_len))]
        # alpha = torch.tensor([random.betavariate(0.4,0.4) for _ in range(x.size(1))]).unsqueeze(0).unsqueeze(-1).type_as(x)
        # mixed_x = alpha * x[:batch_len] + (1-alpha) * pseudo_features
        # x = torch.cat([x[:batch_len], img, mixed_x], dim=0)
        x = x[:batch_len] + 0.4*pseudo_features*layer_idx

        return x

    def forward(self, x, src_img_features, encoder_padding_mask, batch_len, lay_idx,
                attn_mask: Optional[Tensor] = None):

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e4)

        x, _ = self.self_attn(
            query=x,
            key=x[:batch_len],
            value=x[:batch_len],
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        # x = torch.cat([x,x],dim=0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x

        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # src_img_features = self.image_encoder(src_img_features)

        # src_img_features_relate = self.gating(x, src_img_features)
        # multimodal_semantic_att = self.text_image(x, src_img_features, encoder_padding_mask)

        # if lay_idx >= 0:

        # multimodal_semantic_att = self.text_image(x, src_img_features, encoder_padding_mask)

        multimodal_semantic_att = self.text_image(x, src_img_features, encoder_padding_mask, batch_len)

        src_img_features_relate, src_img_features_no_relate = self.gating(x[:batch_len], multimodal_semantic_att)

        # a,b,c = src_img_features_no_relate.size()
        # src_img_features_no_relate = src_img_features_no_relate*0     #图片置0
        # x_interact = x + src_img_features_relate
        # x = torch.cat([x[:batch_len],src_img_features_relate], dim=0)

        # if not self.normalize_before:
        #     x = self.final_layer_norm(x)

        x = self.multimodel_mix(x,src_img_features_relate,batch_len,lay_idx)
        
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        # no_relate 前三层linear，加到最后一层
        # if lay_idx < 3:
        #     src_img_features_no_relate = self.img_fc2(src_img_features_no_relate)
        if lay_idx == 0:
            src_img_features_no_relate = self.img_fc3(src_img_features_no_relate)
        if lay_idx == 1:
            src_img_features_no_relate = self.img_fc4(src_img_features_no_relate)
        if lay_idx == 2:
            src_img_features_no_relate = self.img_fc5(src_img_features_no_relate)
        return x, src_img_features_relate, src_img_features_no_relate


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        ##### img  #######
        self.encoder_attn_img = MultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )
        #####  no help img #########
        self.encoder_attn_img_no = MultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        # self.MLP = Linear(self.embed_dim, self.embed_dim)
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        self.fc_txt_img = Linear(self.embed_dim*2, 1)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


    def forward(
        self,
        x,
            i,
            encoder_states,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

#        if encoder_states is not None:
#            encoder_out = encoder_states[3]
#            i = i + 1

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x_img_no = None
        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            # x, attn = self.encoder_attn(
            #     query=x,
            #     key=encoder_out,
            #     value=encoder_out,
            #     key_padding_mask=encoder_padding_mask,
            #     incremental_state=incremental_state,
            #     static_kv=True,
            #     need_weights=need_attn or (not self.training and self.need_attn),
            #     need_head_weights=need_head_weights,
            # )
            # x = F.dropout(x, p=self.dropout, training=self.training)

            if encoder_states is not None:
                test_ = encoder_states[0]
            # print(test_.size())

            x_tgt = x

            ##########  cross attention src and tgt  #########
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out[:encoder_padding_mask.size(1)],
                value=encoder_out[:encoder_padding_mask.size(1)],
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            x_txt = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            ##########  cross attention help img and tgt  ##########
            # encoder_padding_mask_img = torch.zeros(encoder_out.size(1), 49).eq(1).cuda()
            encoder_padding_mask_img = torch.zeros(encoder_out.size(1), 49).eq(1)
            x_img, attn_1 = self.encoder_attn_img(
                query=x_tgt,
                key=encoder_out[encoder_padding_mask.size(1): encoder_padding_mask.size(1) + 49],
                # key=src_img,
                value=encoder_out[encoder_padding_mask.size(1): encoder_padding_mask.size(1) + 49],
                key_padding_mask=encoder_padding_mask_img,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x_img = F.dropout(x_img, p=self.dropout, training=self.training)

            ##########  cross attention help img and tgt  ##########
            x_img_no_help = encoder_out[encoder_padding_mask.size(1) + 49:]

            x_img_no, attn_2 = self.encoder_attn_img_no(
                query=x_tgt,
                key=x_img_no_help,
                value=x_img_no_help,
                key_padding_mask=None,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x_img_no = F.dropout(x_img_no, p=self.dropout, training=self.training)

            # if attn is not None:
            #     attn = attn + attn_1 + attn_2
            x = x + x_img

            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            ####  融合No help img  #####
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            x_img_no_merge = torch.cat([x, x_img_no],dim=-1)
            x_img_no_merge = torch.sigmoid(self.fc_txt_img(x_img_no_merge))
            x_img_no = torch.mul(x_img_no_merge, x_img_no)
            x = x + x_img_no
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)



            ####################################################
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        return x, x_txt, x_img, x_img_no, attn, None,i
        # return x, x_txt, x_img_no, attn, None, i
    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

####################################
#########  image encoder #########
class TransformerEncoderLayer_image(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention_Image(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        # self.fc_con = Linear(2*self.embed_dim, self.embed_dim)
        self.fc_con_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        # self.highway_net = HighWayNet(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]


    def forward(self, src_img_features, encoder_padding_mask, mask_matrix_tmp, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = src_img_features

        if self.normalize_before:
            src_img_features = self.self_attn_layer_norm(src_img_features)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)


        src_img_features, _ = self.self_attn(
            query=src_img_features,
            key=src_img_features,
            value=src_img_features,
            mask_matrix_tmp = mask_matrix_tmp.cuda(),
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        src_img_features = F.dropout(src_img_features, p=self.dropout, training=self.training)
        src_img_features = residual + src_img_features


        if not self.normalize_before:
            src_img_features = self.self_attn_layer_norm(src_img_features)
        residual = src_img_features
        if self.normalize_before:
            src_img_features = self.final_layer_norm(src_img_features)
        src_img_features = self.activation_fn(self.fc1(src_img_features))
        src_img_features = F.dropout(src_img_features, p=float(self.activation_dropout), training=self.training)
        src_img_features = self.fc2(src_img_features)
        src_img_features = F.dropout(src_img_features, p=self.dropout, training=self.training)
        src_img_features = residual + src_img_features
        if not self.normalize_before:
            src_img_features = self.final_layer_norm(src_img_features)

        return src_img_features


######### gating  ##########
class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_img = Linear(args.gating_dim * 2, 1)

    def forward(self, x, src_img_features):


        x = torch.mean(x, dim=0, keepdim=True)  ## 1*batch*dim
        t, b, c = src_img_features.shape
        x = x.expand(t, b, c)

        merge = torch.cat([x, src_img_features], dim=-1)

        gate = torch.sigmoid(self.fc_img(merge))  # T B C


        # gate = torch.tanh(self.fc_img(merge))

        help_img_features = torch.mul(gate, src_img_features)

        no_help_img_features = torch.mul(1-gate, src_img_features)
        # img_features_2 = torch.mul(gate_2, grid_img_features)

        # img_features = img_features + img_features_2
        return help_img_features, no_help_img_features