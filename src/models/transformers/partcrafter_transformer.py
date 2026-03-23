# Copyright (c) 2025 Yuchen Lin

# This code is based on TripoSG (https://github.com/VAST-AI-Research/TripoSG). Below is the statement from the original repository:

# This code is based on Tencent HunyuanDiT (https://huggingface.co/Tencent-Hunyuan/HunyuanDiT),
# which is licensed under the Tencent Hunyuan Community License Agreement.
# Portions of this code are copied or adapted from HunyuanDiT.
# See the original license below:

# ---- Start of Tencent Hunyuan Community License Agreement ----

# TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT
# Tencent Hunyuan DiT Release Date: 14 May 2024
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION AND IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying any portion or element of the Tencent Hunyuan Works, including via any Hosted Service, You will be deemed to have recognized and accepted the content of this Agreement, which is effective immediately.
# 1.	DEFINITIONS.
# a.	“Acceptable Use Policy” shall mean the policy made available by Tencent as set forth in the Exhibit A.
# b.	“Agreement” shall mean the terms and conditions for use, reproduction, distribution, modification, performance and displaying of Tencent Hunyuan Works or any portion or element thereof set forth herein.
# c.	“Documentation” shall mean the specifications, manuals and documentation for Tencent Hunyuan made publicly available by Tencent.
# d.	“Hosted Service” shall mean a hosted service offered via an application programming interface (API), web access, or any other electronic or remote means.
# e.	“Licensee,” “You” or “Your” shall mean a natural person or legal entity exercising the rights granted by this Agreement and/or using the Tencent Hunyuan Works for any purpose and in any field of use.
# f.	“Materials” shall mean, collectively, Tencent’s proprietary Tencent Hunyuan and Documentation (and any portion thereof) as made available by Tencent under this Agreement.
# g.	“Model Derivatives” shall mean all: (i) modifications to Tencent Hunyuan or any Model Derivative of Tencent Hunyuan; (ii) works based on Tencent Hunyuan or any Model Derivative of Tencent Hunyuan; or (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Tencent Hunyuan or any Model Derivative of Tencent Hunyuan, to that model in order to cause that model to perform similarly to Tencent Hunyuan or a Model Derivative of Tencent Hunyuan, including distillation methods, methods that use intermediate data representations, or methods based on the generation of synthetic data Outputs by Tencent Hunyuan or a Model Derivative of Tencent Hunyuan for training that model. For clarity, Outputs by themselves are not deemed Model Derivatives.
# h.	“Output” shall mean the information and/or content output of Tencent Hunyuan or a Model Derivative that results from operating or otherwise using Tencent Hunyuan or a Model Derivative, including via a Hosted Service.
# i.	“Tencent,” “We” or “Us” shall mean THL A29 Limited.
# j.	“Tencent Hunyuan” shall mean the large language models, text/image/video/audio/3D generation models, and multimodal large language models and their software and algorithms, including trained model weights, parameters (including optimizer states), machine-learning model code, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing made publicly available by Us, including, without limitation to, Tencent Hunyuan DiT released at https://huggingface.co/Tencent-Hunyuan/HunyuanDiT.
# k.	“Tencent Hunyuan Works” shall mean: (i) the Materials; (ii) Model Derivatives; and (iii) all derivative works thereof.
# l.	“Territory” shall mean the worldwide territory, excluding the territory of the European Union.
# m.	“Third Party” or “Third Parties” shall mean individuals or legal entities that are not under common control with Us or You.
# n.	“including” shall mean including but not limited to.
# 2.	GRANT OF RIGHTS.
# We grant You, for the Territory only, a non-exclusive, non-transferable and royalty-free limited license under Tencent’s intellectual property or other rights owned by Us embodied in or utilized by the Materials to use, reproduce, distribute, create derivative works of (including Model Derivatives), and make modifications to the Materials, only in accordance with the terms of this Agreement and the Acceptable Use Policy, and You must not violate (or encourage or permit anyone else to violate) any term of this Agreement or the Acceptable Use Policy.
# 3.	DISTRIBUTION.
# You may, subject to Your compliance with this Agreement, distribute or make available to Third Parties the Tencent Hunyuan Works, exclusively in the Territory, provided that You meet all of the following conditions:
# a.	You must provide all such Third Party recipients of the Tencent Hunyuan Works or products or services using them a copy of this Agreement;
# b.	You must cause any modified files to carry prominent notices stating that You changed the files;
# c.	You are encouraged to: (i) publish at least one technology introduction blogpost or one public statement expressing Your experience of using the Tencent Hunyuan Works; and (ii) mark the products or services developed by using the Tencent Hunyuan Works to indicate that the product/service is “Powered by Tencent Hunyuan”; and
# d.	All distributions to Third Parties (other than through a Hosted Service) must be accompanied by a “Notice” text file that contains the following notice: “Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2024 Tencent. All Rights Reserved. The trademark rights of “Tencent Hunyuan” are owned by Tencent or its affiliate.”
# You may add Your own copyright statement to Your modifications and, except as set forth in this Section and in Section 5, may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Model Derivatives as a whole, provided Your use, reproduction, modification, distribution, performance and display of the work otherwise complies with the terms and conditions of this Agreement (including as regards the Territory). If You receive Tencent Hunyuan Works from a Licensee as part of an integrated end user product, then this Section 3 of this Agreement will not apply to You.
# 4.	ADDITIONAL COMMERCIAL TERMS.
# If, on the Tencent Hunyuan version release date, the monthly active users of all products or services made available by or for Licensee is greater than 100 million monthly active users in the preceding calendar month, You must request a license from Tencent, which Tencent may grant to You in its sole discretion, and You are not authorized to exercise any of the rights under this Agreement unless or until Tencent otherwise expressly grants You such rights.
# 5.	RULES OF USE.
# a.	Your use of the Tencent Hunyuan Works must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Tencent Hunyuan Works, which is hereby incorporated by reference into this Agreement. You must include the use restrictions referenced in these Sections 5(a) and 5(b) as an enforceable provision in any agreement (e.g., license agreement, terms of use, etc.) governing the use and/or distribution of Tencent Hunyuan Works and You must provide notice to subsequent users to whom You distribute that Tencent Hunyuan Works are subject to the use restrictions in these Sections 5(a) and 5(b).
# b.	You must not use the Tencent Hunyuan Works or any Output or results of the Tencent Hunyuan Works to improve any other large language model (other than Tencent Hunyuan or Model Derivatives thereof).
# c.	You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan Works, Output or results of the Tencent Hunyuan Works outside the Territory. Any such use outside the Territory is unlicensed and unauthorized under this Agreement.
# 6.	INTELLECTUAL PROPERTY.
# a.	Subject to Tencent’s ownership of Tencent Hunyuan Works made by or for Tencent and intellectual property rights therein, conditioned upon Your compliance with the terms and conditions of this Agreement, as between You and Tencent, You will be the owner of any derivative works and modifications of the Materials and any Model Derivatives that are made by or for You.
# b.	No trademark licenses are granted under this Agreement, and in connection with the Tencent Hunyuan Works, Licensee may not use any name or mark owned by or associated with Tencent or any of its affiliates, except as required for reasonable and customary use in describing and distributing the Tencent Hunyuan Works. Tencent hereby grants You a license to use “Tencent Hunyuan” (the “Mark”) in the Territory solely as required to comply with the provisions of Section 3(c), provided that You comply with any applicable laws related to trademark protection. All goodwill arising out of Your use of the Mark will inure to the benefit of Tencent.
# c.	If You commence a lawsuit or other proceedings (including a cross-claim or counterclaim in a lawsuit) against Us or any person or entity alleging that the Materials or any Output, or any portion of any of the foregoing, infringe any intellectual property or other right owned or licensable by You, then all licenses granted to You under this Agreement shall terminate as of the date such lawsuit or other proceeding is filed. You will defend, indemnify and hold harmless Us from and against any claim by any Third Party arising out of or related to Your or the Third Party’s use or distribution of the Tencent Hunyuan Works.
# d.	Tencent claims no rights in Outputs You generate. You and Your users are solely responsible for Outputs and their subsequent uses.
# 7.	DISCLAIMERS OF WARRANTY AND LIMITATIONS OF LIABILITY.
# a.	We are not obligated to support, update, provide training for, or develop any further version of the Tencent Hunyuan Works or to grant any license thereto.
# b.	UNLESS AND ONLY TO THE EXTENT REQUIRED BY APPLICABLE LAW, THE TENCENT HUNYUAN WORKS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED “AS IS” WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES OF ANY KIND INCLUDING ANY WARRANTIES OF TITLE, MERCHANTABILITY, NONINFRINGEMENT, COURSE OF DEALING, USAGE OF TRADE, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING, REPRODUCING, MODIFYING, PERFORMING, DISPLAYING OR DISTRIBUTING ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS AND ASSUME ANY AND ALL RISKS ASSOCIATED WITH YOUR OR A THIRD PARTY’S USE OR DISTRIBUTION OF ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS AND YOUR EXERCISE OF RIGHTS AND PERMISSIONS UNDER THIS AGREEMENT.
# c.	TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL TENCENT OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, FOR ANY DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY, CONSEQUENTIAL OR PUNITIVE DAMAGES, OR LOST PROFITS OF ANY KIND ARISING FROM THIS AGREEMENT OR RELATED TO ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS, EVEN IF TENCENT OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.
# 8.	SURVIVAL AND TERMINATION.
# a.	The term of this Agreement shall commence upon Your acceptance of this Agreement or access to the Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein.
# b.	We may terminate this Agreement if You breach any of the terms or conditions of this Agreement. Upon termination of this Agreement, You must promptly delete and cease use of the Tencent Hunyuan Works. Sections 6(a), 6(c), 7 and 9 shall survive the termination of this Agreement.
# 9.	GOVERNING LAW AND JURISDICTION.
# a.	This Agreement and any dispute arising out of or relating to it will be governed by the laws of the Hong Kong Special Administrative Region of the People’s Republic of China, without regard to conflict of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement.
# b.	Exclusive jurisdiction and venue for any dispute arising out of or relating to this Agreement will be a court of competent jurisdiction in the Hong Kong Special Administrative Region of the People’s Republic of China, and Tencent and Licensee consent to the exclusive jurisdiction of such court with respect to any such dispute.
#
# EXHIBIT A
# ACCEPTABLE USE POLICY

# Tencent reserves the right to update this Acceptable Use Policy from time to time.
# Last modified: [insert date]

# Tencent endeavors to promote safe and fair use of its tools and features, including Tencent Hunyuan. You agree not to use Tencent Hunyuan or Model Derivatives:
# 1.	Outside the Territory;
# 2.	In any way that violates any applicable national, federal, state, local, international or any other law or regulation;
# 3.	To harm Yourself or others;
# 4.	To repurpose or distribute output from Tencent Hunyuan or any Model Derivatives to harm Yourself or others;
# 5.	To override or circumvent the safety guardrails and safeguards We have put in place;
# 6.	For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# 7.	To generate or disseminate verifiably false information and/or content with the purpose of harming others or influencing elections;
# 8.	To generate or facilitate false online engagement, including fake reviews and other means of fake online engagement;
# 9.	To intentionally defame, disparage or otherwise harass others;
# 10.	To generate and/or disseminate malware (including ransomware) or any other content to be used for the purpose of harming electronic systems;
# 11.	To generate or disseminate personal identifiable information with the purpose of harming others;
# 12.	To generate or disseminate information (including images, code, posts, articles), and place the information in any public context (including –through the use of bot generated tweets), without expressly and conspicuously identifying that the information and/or content is machine generated;
# 13.	To impersonate another individual without consent, authorization, or legal right;
# 14.	To make high-stakes automated decisions in domains that affect an individual’s safety, rights or wellbeing (e.g., law enforcement, migration, medicine/health, management of critical infrastructure, safety components of products, essential services, credit, employment, housing, education, social scoring, or insurance);
# 15.	In a manner that violates or disrespects the social ethics and moral standards of other countries or regions;
# 16.	To perform, facilitate, threaten, incite, plan, promote or encourage violent extremism or terrorism;
# 17.	For any use intended to discriminate against or harm individuals or groups based on protected characteristics or categories, online or offline social behavior or known or predicted personal or personality characteristics;
# 18.	To intentionally exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# 19.	For military purposes;
# 20.	To engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or other professional practices.

# ---- End of Tencent Hunyuan Community License Agreement ----

# Please note that the use of this code is subject to the terms and conditions
# of the Tencent Hunyuan Community License Agreement, including the Acceptable Use Policy.

from typing import *

from collections import defaultdict
import torch
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    FP32LayerNorm,
    LayerNorm,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn

from ..attention_processor import (
    FusedTripoSGAttnProcessor2_0,
    TripoSGAttnProcessor2_0,
    PartCrafterAttnProcessor,
    PartFrameCrafterAttnProcessor,
)
from .modeling_outputs import Transformer1DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class DiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        self_attention_norm_type: Optional[str] = None, 
        use_cross_attention: bool = True, # ada layer norm
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = "fp32_layer_norm",
        dropout=0.0,
        activation_fn: str = "gelu",
        norm_type: str = "fp32_layer_norm",  # TODO
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,  # int(dim * 4) if None
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,  # [x, skip] or [skip, x]
        skip_norm_last: bool = False,  # this is an error
        qk_norm: bool = True,
        qkv_bias: bool = True,
        block_id: Optional[int] = None,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last
        self.block_id = block_id
        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        if use_self_attention:
            if (
                self_attention_norm_type == "fp32_layer_norm"
                or self_attention_norm_type is None
            ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                raise NotImplementedError

            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2. Cross-Attn
        if use_cross_attention:
            assert cross_attention_dim is not None

            self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_topk(self, topk):
        self.flash_processor.topk = topk

    def set_flash_processor(self, flash_processor):
        self.flash_processor = flash_processor
        self.attn2.processor = self.flash_processor

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Prepare attention kwargs
        attention_kwargs = attention_kwargs or {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None and skip is not None:
            cat = torch.cat(
                (
                    [skip, hidden_states]
                    if self.skip_concat_front
                    else [hidden_states, skip]
                ),
                dim=-1,
            )
            if self.skip_norm_last:
                # don't do this
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)
        elif self.skip_linear is not None and skip is None:
            # Skip connection requested but no tensor provided; fall back to identity.
            pass

        # 1. Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        if self.use_cross_attention:
            hidden_states = hidden_states + self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states

# Modified from https://github.com/VAST-AI-Research/TripoSG/blob/main/triposg/models/transformers/triposg_transformer.py#L365
class PartFrameCrafterDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    TripoSG: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
        use_style_cond_and_image_meta_size (`bool`,  *optional*):
            Whether or not to use style condition and image meta size. True for version <=1.1, False for version >= 1.2
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        max_num_parts: int = 32,
        max_num_frames: int = 64,
        enable_part_embedding: bool = True,
        enable_frame_embedding: bool = True,
        enable_local_cross_attn: bool = True,
        enable_global_cross_attn: bool = True,
        enable_static_embedding: bool = False,
        enable_dynamic_embedding: bool = False,
        enable_static_embedding_per_block: bool = False,
        enable_dynamic_embedding_per_block: bool = False,
        global_attn_block_ids: Optional[List[int]] = None,
        global_attn_block_id_range: Optional[List[int]] = None,
        spatial_global_attn_block_ids: Optional[List[int]] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        temporal_global_attn_block_ids: Optional[List[int]] = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    ):
        super().__init__()

        if spatial_global_attn_block_ids is None:
            spatial_global_attn_block_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        if temporal_global_attn_block_ids is None:
            temporal_global_attn_block_ids = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

        print("Initializing PartFrameCrafterDiTModel: ", 
              "num_attention_heads=", num_attention_heads,
              "width=", width,
              "in_channels=", in_channels,
              "num_layers=", num_layers,
              "cross_attention_dim=", cross_attention_dim,
              "max_num_parts=", max_num_parts,
              "max_num_frames=", max_num_frames,
              "enable_part_embedding=", enable_part_embedding,
              "enable_frame_embedding=", enable_frame_embedding,
                "enable_static_embedding=", enable_static_embedding,
                "enable_dynamic_embedding=", enable_dynamic_embedding,
                "enable_static_embedding_per_block=", enable_static_embedding_per_block,
                "enable_dynamic_embedding_per_block=", enable_dynamic_embedding_per_block,
              "enable_local_cross_attn=", enable_local_cross_attn,
              "enable_global_cross_attn=", enable_global_cross_attn,
              "global_attn_block_ids=", global_attn_block_ids,
              "global_attn_block_id_range=", global_attn_block_id_range,
              "spatial_global_attn_block_ids=", spatial_global_attn_block_ids,
              "temporal_global_attn_block_ids=", temporal_global_attn_block_ids,
        )
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )

        if enable_part_embedding:
            print("Using part embedding with max_num_parts =", max_num_parts)
            self.part_embedding = nn.Embedding(max_num_parts, self.inner_dim)
            self.part_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_part_embedding = enable_part_embedding

        if enable_frame_embedding:
            print("Using frame embedding with max_num_frames =", max_num_frames)
            self.frame_embedding = nn.Embedding(max_num_frames, self.inner_dim)
            self.frame_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_frame_embedding = enable_frame_embedding

        if enable_static_embedding and not enable_static_embedding_per_block:
            print("Using static embedding")
            self.static_embedding = nn.Embedding(1, self.inner_dim)
            self.static_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_static_embedding = enable_static_embedding

        if enable_dynamic_embedding and not enable_dynamic_embedding_per_block:
            print("Using dynamic embedding")
            self.dynamic_embedding = nn.Embedding(1, self.inner_dim)
            self.dynamic_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_dynamic_embedding = enable_dynamic_embedding

        if enable_static_embedding and enable_static_embedding_per_block:
            print("Using static embedding per block")
            self.static_embedding_per_block = nn.Embedding(num_layers, self.inner_dim)
            self.static_embedding_per_block.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_static_embedding_per_block = enable_static_embedding_per_block

        if enable_dynamic_embedding and enable_dynamic_embedding_per_block:
            print("Using dynamic embedding per block")
            self.dynamic_embedding_per_block = nn.Embedding(num_layers, self.inner_dim)
            self.dynamic_embedding_per_block.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_dynamic_embedding_per_block = enable_dynamic_embedding_per_block

        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_norm_type=None,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",  # TODO
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,  # this is an error
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    qkv_bias=False,
                    block_id=layer,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.enable_local_cross_attn = enable_local_cross_attn
        self.enable_global_cross_attn = enable_global_cross_attn

        if spatial_global_attn_block_ids is None:
            spatial_global_attn_block_ids = []
        if temporal_global_attn_block_ids is None:
            temporal_global_attn_block_ids = []
        if (not spatial_global_attn_block_ids) and (global_attn_block_ids is not None):
            spatial_global_attn_block_ids = list(global_attn_block_ids)
        if (not spatial_global_attn_block_ids) and (global_attn_block_id_range is not None):
            spatial_global_attn_block_ids = list(range(global_attn_block_id_range[0], global_attn_block_id_range[1] + 1))
        
        self.spatial_global_attn_block_ids = spatial_global_attn_block_ids
        self.temporal_global_attn_block_ids = temporal_global_attn_block_ids

        self.global_attn_block_ids = []
        self.num_layers = num_layers

    def _remove_static_dynamic_embedding(self):
        print("!!! Removing static and dynamic embeddings...")
        if self.enable_static_embedding:
            self.static_embedding.weight.data.zero_()
            self.static_embedding.bias = None
        if self.enable_dynamic_embedding:
            self.dynamic_embedding.weight.data.zero_()
            self.dynamic_embedding.bias = None

    def _set_gradient_checkpointing(
        self, 
        enable: bool = False, 
        gradient_checkpointing_func: Optional[Callable] = None,
    ):
        # TODO: implement gradient checkpointing
        self.gradient_checkpointing = enable

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def set_global_attn_block_ids(self, global_attn_block_ids: List[int]):
        """
        Set the block ids to apply global attention.
        """
        self.global_attn_block_ids = global_attn_block_ids
        print("Set global attention block ids to: ", self.global_attn_block_ids)

        if len(global_attn_block_ids) > 0:
            # Override self-attention processors for global attention blocks
            attn_processor_dict = {}
            modified_attn_processor = []
            for layer_id in range(self.num_layers):
                for attn_id in [1, 2]:
                    if layer_id in global_attn_block_ids:
                        # apply to both self-attention and cross-attention
                        attn_processor_dict[f'blocks.{layer_id}.attn{attn_id}.processor'] = PartFrameCrafterAttnProcessor()
                    else:
                        attn_processor_dict[f'blocks.{layer_id}.attn{attn_id}.processor'] = TripoSGAttnProcessor2_0()
            self.set_attn_processor(attn_processor_dict)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        force_add_static_embedding: bool = False,
        force_add_dynamic_embedding: bool = False,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        _, T, _ = hidden_states.shape

        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        hidden_states = self.proj_in(hidden_states)

        # T + 1 token
        hidden_states = torch.cat([temb, hidden_states], dim=1) # (N, T+1, D)

        # Add part or frame embedding depending on kwargs
        if attention_kwargs is None:
            attention_kwargs = {}
        num_frames_kw = attention_kwargs.get("num_frames", None)
        num_parts_kw = attention_kwargs.get("num_parts", None)
        use_frame_embed = self.enable_frame_embedding and (num_frames_kw is not None) and (
            (isinstance(num_frames_kw, int) and num_frames_kw > 1) or (isinstance(num_frames_kw, torch.Tensor) and (num_frames_kw > 1).any())
        )
        use_part_embed = self.enable_part_embedding and (num_parts_kw is not None) and (
            isinstance(num_parts_kw, int) and num_parts_kw > 1) or (isinstance(num_parts_kw, torch.Tensor) and (num_parts_kw > 1).any()
        )
        used_embed = None
        if use_frame_embed:
            if isinstance(num_frames_kw, torch.Tensor):
                embs = []
                for nf in num_frames_kw:
                    e = self.frame_embedding(torch.arange(int(nf.item()), device=hidden_states.device))
                    embs.append(e)
                used_embed = torch.cat(embs, dim=0)
            elif isinstance(num_frames_kw, int):
                used_embed = self.frame_embedding(torch.arange(hidden_states.shape[0], device=hidden_states.device))

        elif use_part_embed:
            if isinstance(num_parts_kw, torch.Tensor):
                embs = []
                for npv in num_parts_kw:
                    e = self.part_embedding(torch.arange(int(npv.item()), device=hidden_states.device))
                    embs.append(e)
                used_embed = torch.cat(embs, dim=0)
            elif isinstance(num_parts_kw, int):
                used_embed = self.part_embedding(torch.arange(hidden_states.shape[0], device=hidden_states.device))

        if used_embed is not None:
            hidden_states = hidden_states + used_embed.unsqueeze(dim=1)

        # Add static or dynamic embedding depending on kwargs
        use_static_embed = self.enable_static_embedding and (num_parts_kw is not None) and (
            isinstance(num_parts_kw, int) and num_parts_kw > 1) or (isinstance(num_parts_kw, torch.Tensor) and (num_parts_kw > 1).any()
        )
        use_dynamic_embed = self.enable_dynamic_embedding and (num_frames_kw is not None) and (
            (isinstance(num_frames_kw, int) and num_frames_kw > 1) or (isinstance(num_frames_kw, torch.Tensor) and (num_frames_kw > 1).any())
        ) 

        if force_add_static_embedding or (self.enable_static_embedding and use_static_embed and not self.enable_static_embedding_per_block):
            # print("Adding static embedding")
            static_embed = self.static_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states.device))
            hidden_states = hidden_states + static_embed.unsqueeze(dim=1)
        elif force_add_dynamic_embedding or (self.enable_dynamic_embedding and use_dynamic_embed and not self.enable_dynamic_embedding_per_block):
            # print("Adding dynamic embedding")
            dynamic_embed = self.dynamic_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states.device))
            hidden_states = hidden_states + dynamic_embed.unsqueeze(dim=1)

        # prepare negative encoder_hidden_states
        negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states) if encoder_hidden_states is not None else None

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()
            if (
                (not self.enable_local_cross_attn)
                and (layer not in self.global_attn_block_ids)
            ):
                # If in non-global attention block and disable local cross attention, use negative encoder_hidden_states
                # Do not inject control signal into non-global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            elif (
                (not self.enable_global_cross_attn)
                and (layer in self.global_attn_block_ids)
            ):
                # If in global attention block and disable global cross attention, use negative encoder_hidden_states
                # Do not inject control signal into global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            else:
                input_encoder_hidden_states = encoder_hidden_states
            
            if len(self.global_attn_block_ids) > 0 and (layer in self.global_attn_block_ids):
                # Inject control signal into global attention block
                input_attention_kwargs = attention_kwargs
            else:
                input_attention_kwargs = None

            ### Use per-block static or dynamic embedding if applicable
            if use_static_embed and self.enable_static_embedding_per_block:
                static_embed = self.static_embedding_per_block(torch.tensor([layer] * hidden_states.shape[0], device=hidden_states.device))
                hidden_states = hidden_states + static_embed.unsqueeze(dim=1)
            elif use_dynamic_embed and self.enable_dynamic_embedding_per_block:
                dynamic_embed = self.dynamic_embedding_per_block(torch.tensor([layer] * hidden_states.shape[0], device=hidden_states.device))
                hidden_states = hidden_states + dynamic_embed.unsqueeze(dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    input_encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    skip,
                    input_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=input_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=input_attention_kwargs,
                )  # (N, T+1, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -T:]  # (N, T, D)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)
    
    def forward_1(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_masked: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        force_add_static_embedding: bool = False,
        force_add_dynamic_embedding: bool = False,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        _, T, _ = hidden_states.shape

        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        hidden_states = self.proj_in(hidden_states)

        # T + 1 token
        hidden_states = torch.cat([temb, hidden_states], dim=1) # (N, T+1, D)

        # Add part or frame embedding depending on kwargs
        if attention_kwargs is None:
            attention_kwargs = {}
        num_frames_kw = attention_kwargs.get("num_frames", None)
        num_parts_kw = attention_kwargs.get("num_parts", None)
        use_frame_embed = self.enable_frame_embedding and (num_frames_kw is not None) and (
            (isinstance(num_frames_kw, int) and num_frames_kw > 1) or (isinstance(num_frames_kw, torch.Tensor) and (num_frames_kw > 1).any())
        )
        use_part_embed = self.enable_part_embedding and (num_parts_kw is not None) and (
            isinstance(num_parts_kw, int) and num_parts_kw > 1) or (isinstance(num_parts_kw, torch.Tensor) and (num_parts_kw > 1).any()
        )
        used_embed = None
        if use_frame_embed:
            if isinstance(num_frames_kw, torch.Tensor):
                embs = []
                for nf in num_frames_kw:
                    e = self.frame_embedding(torch.arange(int(nf.item()), device=hidden_states.device))
                    embs.append(e)
                used_embed = torch.cat(embs, dim=0)
            elif isinstance(num_frames_kw, int):
                used_embed = self.frame_embedding(torch.arange(hidden_states.shape[0], device=hidden_states.device))

        elif use_part_embed:
            if isinstance(num_parts_kw, torch.Tensor):
                embs = []
                for npv in num_parts_kw:
                    e = self.part_embedding(torch.arange(int(npv.item()), device=hidden_states.device))
                    embs.append(e)
                used_embed = torch.cat(embs, dim=0)
            elif isinstance(num_parts_kw, int):
                used_embed = self.part_embedding(torch.arange(hidden_states.shape[0], device=hidden_states.device))

        if used_embed is not None:
            hidden_states = hidden_states + used_embed.unsqueeze(dim=1)

        # Add static or dynamic embedding depending on kwargs
        use_static_embed = self.enable_static_embedding and (num_parts_kw is not None) and (
            isinstance(num_parts_kw, int) and num_parts_kw > 1) or (isinstance(num_parts_kw, torch.Tensor) and (num_parts_kw > 1).any()
        )
        use_dynamic_embed = self.enable_dynamic_embedding and (num_frames_kw is not None) and (
            (isinstance(num_frames_kw, int) and num_frames_kw > 1) or (isinstance(num_frames_kw, torch.Tensor) and (num_frames_kw > 1).any())
        ) 

        if force_add_static_embedding or (self.enable_static_embedding and use_static_embed and not self.enable_static_embedding_per_block):
            # print("Adding static embedding")
            static_embed = self.static_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states.device))
            hidden_states = hidden_states + static_embed.unsqueeze(dim=1)
        elif force_add_dynamic_embedding or (self.enable_dynamic_embedding and use_dynamic_embed and not self.enable_dynamic_embedding_per_block):
            # print("Adding dynamic embedding")
            dynamic_embed = self.dynamic_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states.device))
            hidden_states = hidden_states + dynamic_embed.unsqueeze(dim=1)
        # prepare negative encoder_hidden_states
        negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states) if encoder_hidden_states is not None else None

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()
            if (
                (not self.enable_local_cross_attn)
                and (layer not in self.global_attn_block_ids)
            ):
                # If in non-global attention block and disable local cross attention, use negative encoder_hidden_states
                # Do not inject control signal into non-global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            elif (
                (not self.enable_global_cross_attn)
                and (layer in self.global_attn_block_ids)
            ):
                # If in global attention block and disable global cross attention, use negative encoder_hidden_states
                # Do not inject control signal into global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            else:
                if layer in self.spatial_global_attn_block_ids:
                    input_encoder_hidden_states = encoder_hidden_states
                else:
                    input_encoder_hidden_states = encoder_hidden_states_masked
            
            if len(self.global_attn_block_ids) > 0 and (layer in self.global_attn_block_ids):
                # Inject control signal into global attention block
                input_attention_kwargs = attention_kwargs
            else:
                input_attention_kwargs = None

            ### Use per-block static or dynamic embedding if applicable
            if use_static_embed and self.enable_static_embedding_per_block:
                static_embed = self.static_embedding_per_block(torch.tensor([layer] * hidden_states.shape[0], device=hidden_states.device))
                hidden_states = hidden_states + static_embed.unsqueeze(dim=1)
            elif use_dynamic_embed and self.enable_dynamic_embedding_per_block:
                dynamic_embed = self.dynamic_embedding_per_block(torch.tensor([layer] * hidden_states.shape[0], device=hidden_states.device))
                hidden_states = hidden_states + dynamic_embed.unsqueeze(dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    input_encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    skip,
                    input_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=input_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=input_attention_kwargs,
                )  # (N, T+1, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -T:]  # (N, T, D)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    def set_block_attn_processor(self, block_id: int, processor: AttentionProcessor):
        """
        Sets the attention processor for a specific attention block.

        Parameters:
            block_id (`int`):
                The block id of the attention processor to set.
            attn_id (`int`):
                The attention id of the attention processor to set. Either 1 or 2.
            processor (`AttentionProcessor`):
                The instantiated processor class to set as the processor.
        """ 

        self.blocks[block_id].attn1.set_processor(processor)
        self.blocks[block_id].attn2.set_processor(processor)

    def forward_matrix(
        self,
        hidden_states_matrix: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states_matrix_temporal: Optional[torch.Tensor] = None,
        encoder_hidden_states_matrix_spatial: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        static_count: Optional[int] = None,
        dynamic_count: Optional[int] = None,
        return_dict: bool = True,
        cutoff: Optional[bool] = False,
    ):
        # M: Spatial dimension, e.g. number of parts
        # N: Temporal dimension, e.g. number of frames
        M, N, T, _ = hidden_states_matrix.shape

        temb = self.time_embed(timestep).to(hidden_states_matrix.dtype)

        temb = self.time_proj(temb)

        temb = temb.unsqueeze(dim=1).unsqueeze(dim=1).repeat(M, N, 1, 1)  # unsqueeze to concat with hidden_states

        # print("temb shape:", temb.shape, "hidden_states_matrix shape before proj_in:", hidden_states_matrix.shape)

        hidden_states_matrix_ = hidden_states_matrix.clone()
        hidden_states_matrix = torch.zeros((M, N, T, self.inner_dim), dtype=hidden_states_matrix_.dtype, device=hidden_states_matrix_.device)

        for i in range(M):
            hidden_states_matrix[i] = self.proj_in(hidden_states_matrix_[i])

        hidden_states_matrix = torch.cat([temb, hidden_states_matrix], dim=2) # (M, N, T+1, D)

        # print("hidden_states_matrix shape after proj_in and adding temb:", hidden_states_matrix.shape)

        if self.enable_frame_embedding and N > 1:
            used_embed = self.frame_embedding(torch.arange(N, device=hidden_states_matrix.device))
            used_embed = used_embed.unsqueeze(dim=0).repeat(M, 1, 1)

            hidden_states_matrix = hidden_states_matrix + used_embed.unsqueeze(dim=2)
            
        if self.enable_part_embedding and M > 1:
            used_embed = self.part_embedding(torch.arange(M, device=hidden_states_matrix.device))
            used_embed = used_embed.unsqueeze(dim=1).repeat(1, N, 1)

            hidden_states_matrix = hidden_states_matrix + used_embed.unsqueeze(dim=2)

        # hidden_states_matrix shape: (static_count, dynamic_count, T+1, D)
        # hidden_states_matrix shape: (M, N, T+1, D)

        if self.enable_static_embedding:
            static_embed = self.static_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states_matrix.device))
            emb = static_embed.unsqueeze(dim=0).unsqueeze(dim=0)
            hidden_states_matrix[:static_count, :, :, :] = hidden_states_matrix[:static_count, :, :, :] + emb
        if self.enable_dynamic_embedding:
            dynamic_embed = self.dynamic_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states_matrix.device))
            emb = dynamic_embed.unsqueeze(dim=0).unsqueeze(dim=0)
            hidden_states_matrix[static_count:, :, :, :] = hidden_states_matrix[static_count:, :, :, :] + emb

        skips = defaultdict(list)

        for layer, block in enumerate(self.blocks):
            self.set_block_attn_processor(layer, PartFrameCrafterAttnProcessor())

            if layer in self.spatial_global_attn_block_ids:

                for col in range(N):
                    hidden_states = hidden_states_matrix[:, col, :, :]  # (M, T+1, D)
                    encoder_hidden_states = encoder_hidden_states_matrix_spatial[:, col, :, :]
                    
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb[:, col, :, :],
                        image_rotary_emb=image_rotary_emb,
                        skip=None if layer <= self.config.num_layers // 2 else skips[f"spatial_{col}"].pop(),
                        attention_kwargs={"num_parts": M if not cutoff else None, "num_frames": None},
                    )

                    if layer < self.config.num_layers // 2:
                        skips[f"spatial_{col}"].append(hidden_states)

                    hidden_states_matrix[:, col, :, :] = hidden_states

                continue

            # temporal block
            for row in range(M):
                hidden_states = hidden_states_matrix[row, :, :, :]  # (N, T+1, D)
                encoder_hidden_states = encoder_hidden_states_matrix_temporal[row, :, :, :]

                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb[row, :, :, :],
                    image_rotary_emb=image_rotary_emb,
                    skip=None if layer <= self.config.num_layers // 2 else skips[f"temporal_{row}"].pop(),
                    attention_kwargs={"num_parts": None, "num_frames": (None if row < static_count else N)},
                )

                if layer < self.config.num_layers // 2:
                    skips[f"temporal_{row}"].append(hidden_states)

                hidden_states_matrix[row, :, :, :] = hidden_states    

        hidden_states_matrix_ = hidden_states_matrix.clone()
        hidden_states_matrix = torch.zeros((M, N, T, self.out_channels), dtype=hidden_states_matrix_.dtype, device=hidden_states_matrix_.device)

        # final layer
        for i in range(M):
            hidden_states_matrix_[i] = self.norm_out(hidden_states_matrix_[i])
            hidden_states_matrix[i] = self.proj_out(hidden_states_matrix_[i][:, -T:])

        if not return_dict:
            return (hidden_states_matrix,)

        return Transformer1DModelOutput(sample=hidden_states_matrix)

    def forward_spatiotemporal(
        self,
        hidden_states_static: Optional[torch.Tensor],
        hidden_states_dynamic: Optional[torch.Tensor],
        timestep_static: Union[int, float, torch.LongTensor],
        timestep_dynamic: Union[int, float, torch.LongTensor],
        encoder_hidden_states_static: Optional[torch.Tensor] = None,
        encoder_hidden_states_dynamic: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        static_count: Optional[int] = None,
        dynamic_count: Optional[int] = None,
        return_dict: bool = True,
    ):
        # print("Forwarding PartFrameCrafterDiTModel with spatiotemporal inputs", static_count, dynamic_count)
        # print("hidden_states_static shape:", hidden_states_static.shape)
        # print("hidden_states_dynamic shape:", hidden_states_dynamic.shape)
        # print("encoder_hidden_states_static shape:", encoder_hidden_states_static.shape if encoder_hidden_states_static
        #         is not None else None)
        # print("encoder_hidden_states_dynamic shape:", encoder_hidden_states_dynamic.shape if encoder_hidden_states_dynamic
        #         is not None else None)
        
        _, T, _ = hidden_states_static.shape

        temb_static = self.time_embed(timestep_static).to(hidden_states_static.dtype)
        temb_dynamic = self.time_embed(timestep_dynamic).to(hidden_states_dynamic.dtype)

        temb_static = self.time_proj(temb_static)
        temb_static = temb_static.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        temb_dynamic = self.time_proj(temb_dynamic)
        temb_dynamic = temb_dynamic.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        hidden_states_static = self.proj_in(hidden_states_static)
        hidden_states_dynamic = self.proj_in(hidden_states_dynamic)

        hidden_states_static = torch.cat([temb_static, hidden_states_static], dim=1)  # (N, T+1, D)
        hidden_states_dynamic = torch.cat([temb_dynamic, hidden_states_dynamic], dim=1)  # (N, T+1, D)

        # Add part or frame embedding depending on kwargs
        num_frames_kw = dynamic_count
        num_parts_kw = static_count

        if num_frames_kw is not None and num_frames_kw > 1 and self.enable_frame_embedding:
            if isinstance(num_frames_kw, torch.Tensor):
                embs = []
                for nf in num_frames_kw:
                    e = self.frame_embedding(torch.arange(int(nf.item()), device=hidden_states_dynamic.device))
                    embs.append(e)
                used_embed = torch.cat(embs, dim=0)
            elif isinstance(num_frames_kw, int):
                used_embed = self.frame_embedding(torch.arange(hidden_states_dynamic.shape[0], device=hidden_states_dynamic.device))

            # print("Adding frame embedding", used_embed.shape)
            hidden_states_dynamic = hidden_states_dynamic + used_embed.unsqueeze(dim=1)
            
        if self.enable_part_embedding:
            # print("Adding part embedding for static and dynamic inputs", static_count, dynamic_count)
            static_embedding = self.part_embedding(torch.arange(static_count, device=hidden_states_static.device))
            static_embedding_for_dynamic = self.part_embedding(torch.tensor([static_count] * dynamic_count, device=hidden_states_dynamic.device))

            hidden_states_static = hidden_states_static + static_embedding.unsqueeze(dim=1)
            hidden_states_dynamic = hidden_states_dynamic + static_embedding_for_dynamic.unsqueeze(dim=1)

        # Add static or dynamic embedding depending on kwargs
        if not self.enable_static_embedding_per_block and self.enable_static_embedding:
            static_embed = self.static_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states_static.device))
            hidden_states_static = hidden_states_static + static_embed.unsqueeze(dim=1)

        if not self.enable_dynamic_embedding_per_block and self.enable_dynamic_embedding:
            dynamic_embed = self.dynamic_embedding(torch.zeros(1, dtype=torch.long, device=hidden_states_dynamic.device))
            hidden_states_dynamic = hidden_states_dynamic + dynamic_embed.unsqueeze(dim=1)

        skips = defaultdict(list)

        for layer, block in enumerate(self.blocks):
            
            if layer in self.spatial_global_attn_block_ids:
                # print("Spatial global attention block:", layer)
                # Set spatial global attention processor
                self.set_block_attn_processor(layer, PartFrameCrafterAttnProcessor())
                
                if self.enable_static_embedding_per_block:
                    # print("Adding static embedding per block at layer", layer)
                    static_embed = self.static_embedding_per_block(torch.tensor([layer] * hidden_states_static.shape[0], device=hidden_states_static.device))
                    hidden_states_static = hidden_states_static + static_embed.unsqueeze(dim=1)

                if self.enable_dynamic_embedding_per_block:
                    # print("Adding dynamic embedding per block at layer", layer)
                    dynamic_embed = self.dynamic_embedding_per_block(torch.tensor([layer] * hidden_states_dynamic.shape[0], device=hidden_states_dynamic.device))
                    hidden_states_dynamic = hidden_states_dynamic + dynamic_embed.unsqueeze(dim=1)

                skip_values = []
    
                for hidden_state_dynamic_index, hidden_state_dynamic in enumerate(hidden_states_dynamic):
                    hidden_state_dynamic_input = hidden_state_dynamic.unsqueeze(dim=0)
                    
                    hidden_states_input = torch.cat([hidden_states_static.clone(), hidden_state_dynamic_input], dim=0)
                    # encoder_hidden_states_input = encoder_hidden_states_dynamic[hidden_state_dynamic_index].unsqueeze(dim=0).repeat(hidden_states_input.shape[0], 1, 1)
                    encoder_hidden_states_input = torch.cat([encoder_hidden_states_static[hidden_state_dynamic_index].unsqueeze(dim=0).repeat(static_count, 1, 1), encoder_hidden_states_dynamic[hidden_state_dynamic_index].unsqueeze(0)], dim=0)
                    # encoder_hidden_states_input = encoder_hidden_states_static[hidden_state_dynamic_index].unsqueeze(dim=0).repeat(static_count + 1, 1, 1)
                
                    new_hidden_states = block(
                        hidden_states_input,
                        encoder_hidden_states=encoder_hidden_states_input,
                        temb=temb_static[:1].repeat(hidden_states_input.shape[0], 1, 1),
                        image_rotary_emb=image_rotary_emb,
                        skip=None if layer <= self.config.num_layers // 2 else skips[f"spatial_block__dynamic_{hidden_state_dynamic_index}"].pop(),
                        attention_kwargs={"num_parts": static_count + 1, "num_frames": 1}
                    )  # (N, T+1, D)

                    if layer < self.config.num_layers // 2:
                        skips[f"spatial_block__dynamic_{hidden_state_dynamic_index}"].append(new_hidden_states)

                    hidden_states_dynamic[hidden_state_dynamic_index] = new_hidden_states[-1]

                    if hidden_state_dynamic_index == dynamic_count - 1:
                        hidden_states_static = new_hidden_states[:static_count]
                
                self.set_block_attn_processor(layer, TripoSGAttnProcessor2_0())

            elif layer in self.temporal_global_attn_block_ids:
                # print("Temporal global attention block:", layer)
                self.set_block_attn_processor(layer, PartFrameCrafterAttnProcessor())

                hidden_states_input = hidden_states_dynamic
                encoder_hidden_states_input = encoder_hidden_states_dynamic

                if self.enable_dynamic_embedding_per_block:
                    # print("Adding dynamic embedding per block at layer", layer)
                    dynamic_embed = self.dynamic_embedding_per_block(torch.tensor([layer] * hidden_states_input.shape[0], device=hidden_states_input.device))
                    hidden_states_input = hidden_states_input + dynamic_embed.unsqueeze(dim=1)

                new_hidden_states_dynamic = block(
                    hidden_states_input,
                    encoder_hidden_states=encoder_hidden_states_input,
                    temb=temb_dynamic[:1].repeat(hidden_states_input.shape[0], 1, 1),
                    image_rotary_emb=image_rotary_emb,
                    skip=None if layer <= self.config.num_layers // 2 else skips["temporal_block_dynamic"].pop(),
                    attention_kwargs={"num_parts": 1, "num_frames": dynamic_count}
                )  # (N, T+1, D)

                if layer < self.config.num_layers // 2:
                    skips["temporal_block_dynamic"].append(new_hidden_states_dynamic)

                hidden_states_dynamic = new_hidden_states_dynamic

                hidden_states_input = hidden_states_static
                encoder_hidden_states_input = encoder_hidden_states_static[:static_count]

                if self.enable_static_embedding_per_block:
                    print("Adding static embedding per block at layer", layer)
                    static_embed = self.static_embedding_per_block(torch.tensor([layer] * hidden_states_input.shape[0], device=hidden_states_input.device))
                    hidden_states_input = hidden_states_input + static_embed.unsqueeze(dim=1)

                self.set_block_attn_processor(layer, TripoSGAttnProcessor2_0())
                
                new_hidden_states_static = block(
                    hidden_states_input,
                    encoder_hidden_states=encoder_hidden_states_input,
                    temb=temb_static[:1].repeat(hidden_states_input.shape[0], 1, 1),
                    image_rotary_emb=image_rotary_emb,
                    skip=None if layer <= self.config.num_layers // 2 else skips["temporal_block_static"].pop(),
                    # attention_kwargs={"num_parts": 1, "num_frames": 1}
                )  # (N, T+1, D)

                if layer < self.config.num_layers // 2:
                    skips["temporal_block_static"].append(new_hidden_states_static)

                hidden_states_static = new_hidden_states_static

        # for layer, block in enumerate(self.blocks):
            
        #     if layer in self.spatial_global_attn_block_ids:
        #         # print("Spatial global attention block:", layer)
        #         # Set spatial global attention processor
        #         self.set_block_attn_processor(layer, PartFrameCrafterAttnProcessor())
                
        #         if self.enable_static_embedding_per_block:
        #             # print("Adding static embedding per block at layer", layer)
        #             static_embed = self.static_embedding_per_block(torch.tensor([layer] * hidden_states_static.shape[0], device=hidden_states_static.device))
        #             hidden_states_static = hidden_states_static + static_embed.unsqueeze(dim=1)

        #         if self.enable_dynamic_embedding_per_block:
        #             # print("Adding dynamic embedding per block at layer", layer)
        #             dynamic_embed = self.dynamic_embedding_per_block(torch.tensor([layer] * hidden_states_dynamic.shape[0], device=hidden_states_dynamic.device))
        #             hidden_states_dynamic = hidden_states_dynamic + dynamic_embed.unsqueeze(dim=1)

        #         hidden_states_input = torch.cat([hidden_states_static.clone(), hidden_states_dynamic.clone()], dim=0)
        #         encoder_hidden_states_input = torch.cat([encoder_hidden_states_static[:1].repeat(static_count, 1, 1), encoder_hidden_states_dynamic], dim=0)

        #         new_hidden_states = block(
        #             hidden_states_input,
        #             encoder_hidden_states=encoder_hidden_states_input,
        #             temb=temb_static[:1].repeat(hidden_states_input.shape[0], 1, 1),
        #             image_rotary_emb=image_rotary_emb,
        #             skip=None if layer <= self.config.num_layers // 2 else skips["spatial_block"].pop(),
        #             attention_kwargs={"num_parts": static_count + dynamic_count, "num_frames": 1}
        #         )  # (N, T+1, D)

        #         if layer < self.config.num_layers // 2:
        #             skips["spatial_block"].append(new_hidden_states)
                
        #         hidden_states_static = new_hidden_states[:static_count]
        #         hidden_states_dynamic = new_hidden_states[static_count:]

        #         self.set_block_attn_processor(layer, TripoSGAttnProcessor2_0())

        #     elif layer in self.temporal_global_attn_block_ids:
        #         # print("Temporal global attention block:", layer)
        #         self.set_block_attn_processor(layer, PartFrameCrafterAttnProcessor())

        #         hidden_states_input = hidden_states_dynamic
        #         encoder_hidden_states_input = encoder_hidden_states_dynamic

        #         if self.enable_dynamic_embedding_per_block:
        #             dynamic_embed = self.dynamic_embedding_per_block(torch.tensor([layer] * hidden_states_input.shape[0], device=hidden_states_input.device))
        #             hidden_states_input = hidden_states_input + dynamic_embed.unsqueeze(dim=1)

        #         new_hidden_states_dynamic = block(
        #             hidden_states_input,
        #             encoder_hidden_states=encoder_hidden_states_input,
        #             temb=temb_dynamic[:1].repeat(hidden_states_input.shape[0], 1, 1),
        #             image_rotary_emb=image_rotary_emb,
        #             skip=None if layer <= self.config.num_layers // 2 else skips["temporal_block_dynamic"].pop(),
        #             attention_kwargs={"num_parts": 1, "num_frames": dynamic_count}
        #         )  # (N, T+1, D)

        #         if layer < self.config.num_layers // 2:
        #             skips["temporal_block_dynamic"].append(new_hidden_states_dynamic)

        #         hidden_states_dynamic = new_hidden_states_dynamic

        #         hidden_states_input = hidden_states_static
        #         encoder_hidden_states_input = encoder_hidden_states_static[:static_count]

        #         if self.enable_static_embedding_per_block:
        #             print("Adding static embedding per block at layer", layer)
        #             static_embed = self.static_embedding_per_block(torch.tensor([layer] * hidden_states_input.shape[0], device=hidden_states_input.device))
        #             hidden_states_input = hidden_states_input + static_embed.unsqueeze(dim=1)

        #         self.set_block_attn_processor(layer, TripoSGAttnProcessor2_0())
                
        #         new_hidden_states_static = block(
        #             hidden_states_input,
        #             encoder_hidden_states=encoder_hidden_states_input,
        #             temb=temb_static[:1].repeat(hidden_states_input.shape[0], 1, 1),
        #             image_rotary_emb=image_rotary_emb,
        #             skip=None if layer <= self.config.num_layers // 2 else skips["temporal_block_static"].pop(),
        #             # attention_kwargs={"num_parts": 1, "num_frames": 1}
        #         )  # (N, T+1, D)

        #         if layer < self.config.num_layers // 2:
        #             skips["temporal_block_static"].append(new_hidden_states_static)

        #         hidden_states_static = new_hidden_states_static


        # final layer
        hidden_states = self.norm_out(hidden_states_dynamic)
        hidden_states = hidden_states[:, -T:]  # (N, T, D)
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    def forward_spatial_temporal_history(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor, torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *,
        static_count: int,
        dynamic_count: int,
        spatial_static_encoder_states: Optional[torch.Tensor] = None,
        spatial_dynamic_encoder_states: Optional[torch.Tensor] = None,
        temporal_history_latents: Optional[torch.Tensor] = None,
        temporal_history_encoder_states: Optional[torch.Tensor] = None,
        temporal_dynamic_encoder_states: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        spatial_attention_kwargs: Optional[Dict[str, Any]] = None,
        temporal_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """Run a transformer pass alternating spatial and temporal conditioning blocks."""

        if dynamic_count <= 0:
            raise ValueError("dynamic_count must be positive")
        if static_count < 0:
            raise ValueError("static_count must be non-negative")

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            attention_kwargs = {}
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif lora_scale != 1.0:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

        device = hidden_states.device
        dtype = hidden_states.dtype
        latent_token_count = hidden_states.shape[1]

        per_branch = static_count + dynamic_count
        if per_branch <= 0:
            raise ValueError("per-branch size must be positive")
        if hidden_states.shape[0] % per_branch != 0:
            raise ValueError(
                f"Batch dimension {hidden_states.shape[0]} incompatible with static_count={static_count} and dynamic_count={dynamic_count}"
            )
        cfg_factor = hidden_states.shape[0] // per_branch

        def _split_branches(tensor: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
            if tensor is None:
                return [None] * cfg_factor
            chunks: List[torch.Tensor] = []
            for idx in range(cfg_factor):
                start = idx * per_branch
                end = start + per_branch
                chunks.append(tensor[start:end].clone())
            return chunks

        temb = self._compute_time_embeddings(timestep, hidden_states.shape[0], dtype, device)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat([temb, hidden_states], dim=1)

        branch_states = _split_branches(hidden_states)
        branch_tem = _split_branches(temb)
        branch_enc = _split_branches(encoder_hidden_states)

        def _prepare_branch_tensor(
            tensor: Optional[torch.Tensor],
            expected_count: int,
        ) -> List[Optional[torch.Tensor]]:
            if tensor is None:
                if expected_count == 0:
                    return [None] * cfg_factor
                raise ValueError("Missing encoder states for required branch dimension")
            if tensor.shape[0] != cfg_factor:
                raise ValueError(
                    f"Expected {cfg_factor} branches, got tensor with shape {tensor.shape}"
                )
            if tensor.shape[1] != expected_count:
                raise ValueError(
                    f"Expected second dimension {expected_count}, got {tensor.shape}"
                )
            return [tensor[idx].to(device=device) for idx in range(cfg_factor)]

        spatial_static_branches = _prepare_branch_tensor(
            spatial_static_encoder_states, static_count
        )
        spatial_dynamic_branches = _prepare_branch_tensor(
            spatial_dynamic_encoder_states, dynamic_count
        )

        history_count = int(temporal_history_latents.shape[0]) if temporal_history_latents is not None else 0
        if history_count > 0:
            history_latents = temporal_history_latents.to(device=device, dtype=dtype)
            history_hidden = self.proj_in(history_latents)
            if torch.is_tensor(timestep):
                base_timestep = timestep.reshape(-1)[0].detach().clone().to(device=device)
            else:
                base_timestep = torch.tensor(timestep, device=device)
            history_tem = self._compute_time_embeddings(base_timestep, history_count, dtype, device)
            history_hidden = torch.cat([history_tem, history_hidden], dim=1)
        else:
            history_hidden = None
            history_tem = None

        temporal_history_branches = _prepare_branch_tensor(
            temporal_history_encoder_states, history_count
        )
        temporal_dynamic_branches = _prepare_branch_tensor(
            temporal_dynamic_encoder_states, dynamic_count
        )

        branch_history_hidden: List[Optional[torch.Tensor]] = [
            history_hidden.clone() if history_hidden is not None else None
            for _ in range(cfg_factor)
        ]

        def _build_block_kwargs(
            block: DiTBlock,
            base: Optional[Dict[str, Any]],
            override: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            if base is None and override is None:
                return None
            merged: Dict[str, Any] = {}
            if base is not None:
                merged.update(base)
            if override is not None:
                merged.update(override)
            processor = getattr(block.attn1, "processor", None)
            if not isinstance(processor, PartFrameCrafterAttnProcessor):
                merged.pop("num_parts", None)
                merged.pop("num_frames", None)
            return merged or None

        def _build_block_kwargs(
            block: DiTBlock,
            base_kwargs: Optional[Dict[str, Any]],
            override_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            if base_kwargs is None and override_kwargs is None:
                return None
            merged: Dict[str, Any] = {}
            if base_kwargs is not None:
                merged.update(base_kwargs)
            if override_kwargs is not None:
                merged.update(override_kwargs)
            processor = getattr(block.attn1, "processor", None)
            if not isinstance(processor, PartFrameCrafterAttnProcessor):
                merged.pop("num_parts", None)
                merged.pop("num_frames", None)
            return merged or None

        branch_skips: List[List[torch.Tensor]] = [[] for _ in range(cfg_factor)]

        spatial_layers = set(getattr(self, "spatial_global_attn_block_ids", []) or [])
        temporal_layers = set(getattr(self, "temporal_global_attn_block_ids", []) or [])
        active_global_ids = spatial_layers | temporal_layers

        def _apply_spatial_block(
            branch_idx: int,
            block: DiTBlock,
            block_attention: Optional[Dict[str, Any]],
            state: torch.Tensor,
            temb_branch: torch.Tensor,
            skip_branch: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if state.shape[0] == 0:
                return state

            print("Applying spatial block on branch", branch_idx, "with static_count =", static_count, "and dynamic_count =", dynamic_count, state.shape)

            static_enc = spatial_static_branches[branch_idx]
            dynamic_enc = spatial_dynamic_branches[branch_idx]
            num_static = static_count
            num_dynamic = dynamic_count

            new_state = state.clone()
            for dyn_idx in range(num_dynamic):
                frame_state = torch.cat(
                    [state[:num_static], state[num_static + dyn_idx : num_static + dyn_idx + 1]], dim=0
                )
                frame_tem = torch.cat(
                    [temb_branch[:num_static], temb_branch[num_static + dyn_idx : num_static + dyn_idx + 1]], dim=0
                )
                if skip_branch is not None:
                    frame_skip = torch.cat(
                        [skip_branch[:num_static], skip_branch[num_static + dyn_idx : num_static + dyn_idx + 1]],
                        dim=0,
                    )
                else:
                    frame_skip = None

                enc_segments: List[torch.Tensor] = []
                if static_enc is not None and num_static > 0:
                    enc_segments.append(static_enc)
                if dynamic_enc is not None:
                    enc_segments.append(dynamic_enc[dyn_idx : dyn_idx + 1])
                block_enc = torch.cat(enc_segments, dim=0) if enc_segments else None

                local_kwargs = {"num_parts": max(1, num_static + 1), "num_frames": 1}
                block_kwargs = _build_block_kwargs(block, block_attention, local_kwargs)

                print("\nSpatial block forward pass:", frame_state.shape, 
                      "\ntemb.shape", frame_tem.shape,
                      "\nskip.shape" if frame_skip is not None else "skip=None",
                      "\nblock_enc.shape" if block_enc is not None else "block_enc=None",
                      "\nblock_kwargs", block_kwargs,
                )
                
                updated_pair = block(
                    frame_state,
                    encoder_hidden_states=block_enc,
                    temb=frame_tem,
                    skip=frame_skip,
                    attention_kwargs=block_kwargs,
                )

                new_state[num_static + dyn_idx : num_static + dyn_idx + 1] = updated_pair[num_static:]

            new_state[:num_static] = state[:num_static]
            return new_state

        def _apply_temporal_block(
            branch_idx: int,
            block: DiTBlock,
            block_attention: Optional[Dict[str, Any]],
            state: torch.Tensor,
            temb_branch: torch.Tensor,
            skip_branch: Optional[torch.Tensor],
        ) -> torch.Tensor:
            new_state = state.clone()

            dynamic_history_enc = temporal_history_branches[branch_idx]
            dynamic_current_enc = temporal_dynamic_branches[branch_idx]

            current_history = branch_history_hidden[branch_idx]
            history_len = current_history.shape[0] if current_history is not None else 0

            if dynamic_count == 0 and history_len == 0:
                return new_state

            segments_state: List[torch.Tensor] = []
            segments_tem: List[torch.Tensor] = []
            segments_enc: List[torch.Tensor] = []

            if current_history is not None and history_len > 0:
                segments_state.append(current_history)
                segments_tem.append(history_tem)
                if dynamic_history_enc is not None:
                    segments_enc.append(dynamic_history_enc)

            if dynamic_count > 0:
                dynamic_state = state[static_count:]
                dynamic_tem = temb_branch[static_count:]
                segments_state.append(dynamic_state)
                segments_tem.append(dynamic_tem)
                if dynamic_current_enc is not None:
                    segments_enc.append(dynamic_current_enc)

            block_state = torch.cat(segments_state, dim=0)
            block_tem = torch.cat(segments_tem, dim=0)
            block_skip = None
            block_enc = torch.cat(segments_enc, dim=0) if segments_enc else None

            local_temporal_kwargs = {"num_frames": max(1, history_len + dynamic_count)}

            block_kwargs = _build_block_kwargs(block, block_attention, local_temporal_kwargs)
            print("\nTemporal block forward pass:", block_state.shape, 
                "\ntemb.shape", block_tem.shape,
                "\nskip.shape" if block_skip is not None else "skip=None",
                "\nblock_enc.shape" if block_enc is not None else "block_enc=None",
                "\nblock_kwargs", block_kwargs,
            )

            updated = block(
                block_state,
                encoder_hidden_states=block_enc,
                temb=block_tem,
                skip=block_skip,
                attention_kwargs=block_kwargs,
            )

            offset = 0
            if history_len > 0:
                updated_history = updated[offset : offset + history_len]
                branch_history_hidden[branch_idx] = updated_history.detach()
                offset += history_len
            else:
                branch_history_hidden[branch_idx] = None

            if dynamic_count > 0:
                updated_dynamic = updated[offset : offset + dynamic_count]
                new_state[static_count:] = updated_dynamic

            new_state[:static_count] = state[:static_count]
            return new_state

        for layer, block in enumerate(self.blocks):
            is_spatial = layer in spatial_layers
            is_temporal = layer in temporal_layers

            block_attention = attention_kwargs if (layer in active_global_ids) else None

            print(f"\n=== Layer {layer} === is_spatial: {is_spatial}, is_temporal: {is_temporal}, block_attention: {'yes' if block_attention is not None else 'no'}")

            for branch_idx in range(cfg_factor):
                branch_state = branch_states[branch_idx]
                branch_temporal = branch_tem[branch_idx]
                skip_tensor = (
                    None
                    if layer <= self.config.num_layers // 2
                    else branch_skips[branch_idx].pop()
                )

                if is_spatial:
                    updated_state = _apply_spatial_block(
                        branch_idx,
                        block,
                        block_attention,
                        branch_state,
                        branch_temporal,
                        skip_tensor,
                    )
                elif is_temporal:
                    updated_state = _apply_temporal_block(
                        branch_idx,
                        block,
                        block_attention,
                        branch_state,
                        branch_temporal,
                        skip_tensor,
                    )
                else:
                    updated_state = block(
                        branch_state,
                        encoder_hidden_states=branch_enc[branch_idx],
                        temb=branch_temporal,
                        skip=skip_tensor,
                        attention_kwargs=block_attention,
                    )

                branch_states[branch_idx] = updated_state
                if layer < self.config.num_layers // 2:
                    branch_skips[branch_idx].append(updated_state.clone())

        hidden_states = torch.cat(branch_states, dim=0)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -latent_token_count:]
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    def _compute_time_embeddings(
        self,
        timestep: Union[int, float, torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.long)
        else:
            timestep = timestep.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep.repeat(batch_size)
        elif timestep.shape[0] != batch_size:
            timestep = timestep.reshape(-1)
            if timestep.shape[0] == 1:
                timestep = timestep.repeat(batch_size)
            elif timestep.shape[0] != batch_size:
                timestep = timestep.expand(batch_size)
        temb = self.time_embed(timestep).to(dtype)
        temb = self.time_proj(temb)
        return temb.unsqueeze(1)

    def forward_with_history_guidance(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor, torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        history_guidance: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        history_guidance = history_guidance or {}

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_tokens, _ = hidden_states.shape

        if not torch.is_tensor(timestep):
            timesteps_tensor = torch.tensor([timestep], device=hidden_states.device, dtype=torch.long)
        else:
            timesteps_tensor = timestep.to(device=hidden_states.device)
        if timesteps_tensor.ndim == 0:
            timesteps_tensor = timesteps_tensor.repeat(batch_size)
        elif timesteps_tensor.shape[0] != batch_size:
            timesteps_tensor = timesteps_tensor.reshape(-1)
            if timesteps_tensor.shape[0] == 1:
                timesteps_tensor = timesteps_tensor.repeat(batch_size)
            elif timesteps_tensor.shape[0] != batch_size:
                timesteps_tensor = timesteps_tensor.expand(batch_size)

        temb = self._compute_time_embeddings(timesteps_tensor, batch_size, hidden_states.dtype, hidden_states.device)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat([temb, hidden_states], dim=1)

        history_mask = history_guidance.get("history_mask")
        history_indices = None
        if history_mask is not None:
            if not torch.is_tensor(history_mask):
                history_mask = torch.tensor(history_mask, device=hidden_states.device, dtype=torch.bool)
            else:
                history_mask = history_mask.to(device=hidden_states.device, dtype=torch.bool)
            history_mask = history_mask.flatten()
            if history_mask.shape[0] != batch_size:
                raise ValueError(f"history_mask must have length {batch_size}, but got {history_mask.shape[0]}")
            history_indices = torch.nonzero(history_mask, as_tuple=False).flatten()
            if history_indices.numel() == 0:
                history_indices = None

        base_history_hidden = None
        base_history_encoder = None
        if history_indices is not None:
            base_history_hidden = hidden_states[history_indices].clone()
            if encoder_hidden_states is not None:
                base_history_encoder = encoder_hidden_states[history_indices].clone()

        spatial_latents = history_guidance.get("spatial_latents")
        spatial_encoder_states = history_guidance.get("spatial_encoder_states")
        temporal_latents = history_guidance.get("temporal_latents")
        temporal_encoder_states = history_guidance.get("temporal_encoder_states")

        spatial_layers = set(history_guidance.get("spatial_layers", []))
        temporal_layers = set(history_guidance.get("temporal_layers", []))

        history_timesteps = None
        if history_indices is not None:
            history_timesteps = timesteps_tensor[history_indices]

        def _select_history(latents_tensor: Optional[torch.Tensor], encoder_tensor: Optional[torch.Tensor]):
            if history_indices is None or latents_tensor is None:
                return None, None
            latents_tensor = latents_tensor.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if latents_tensor.shape[0] != history_indices.numel():
                repeats = (history_indices.numel() + latents_tensor.shape[0] - 1) // latents_tensor.shape[0]
                latents_tensor = latents_tensor.repeat(repeats, 1, 1)[: history_indices.numel()]
            history_states = self.proj_in(latents_tensor)
            temb_hist = self._compute_time_embeddings(history_timesteps, history_indices.numel(), hidden_states.dtype, hidden_states.device)
            history_states = torch.cat([temb_hist, history_states], dim=1)
            history_enc = None
            if encoder_tensor is not None and encoder_hidden_states is not None:
                encoder_tensor = encoder_tensor.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                if encoder_tensor.shape[0] != history_indices.numel():
                    repeats = (history_indices.numel() + encoder_tensor.shape[0] - 1) // encoder_tensor.shape[0]
                    encoder_tensor = encoder_tensor.repeat(repeats, 1, 1)[: history_indices.numel()]
                history_enc = encoder_tensor
            return history_states, history_enc

        spatial_history_states, spatial_history_enc = _select_history(spatial_latents, spatial_encoder_states)
        temporal_history_states, temporal_history_enc = _select_history(temporal_latents, temporal_encoder_states)

        negative_encoder_hidden_states = (
            torch.zeros_like(encoder_hidden_states) if encoder_hidden_states is not None else None
        )
        skips: List[torch.Tensor] = []
        active_global_ids = set(self.global_attn_block_ids) if getattr(self, "global_attn_block_ids", None) is not None else set()
        if len(active_global_ids) == 0 and hasattr(self, "spatial_global_attn_block_ids"):
            active_global_ids = set(self.spatial_global_attn_block_ids) | set(getattr(self, "temporal_global_attn_block_ids", []))

        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()

            if (
                (not self.enable_local_cross_attn)
                and len(active_global_ids) > 0
                and (layer not in active_global_ids)
            ):
                input_encoder_hidden_states = negative_encoder_hidden_states
            elif (
                (not self.enable_global_cross_attn)
                and len(active_global_ids) > 0
                and (layer in active_global_ids)
            ):
                input_encoder_hidden_states = negative_encoder_hidden_states
            else:
                input_encoder_hidden_states = encoder_hidden_states

            if len(active_global_ids) > 0 and (layer in active_global_ids):
                input_attention_kwargs = attention_kwargs
            else:
                input_attention_kwargs = None

            block_hidden_states = hidden_states
            block_encoder_hidden_states = input_encoder_hidden_states
            if history_indices is not None:
                if layer in spatial_layers and spatial_history_states is not None:
                    block_hidden_states = hidden_states.clone()
                    block_hidden_states[history_indices] = spatial_history_states
                    if block_encoder_hidden_states is not None and spatial_history_enc is not None:
                        block_encoder_hidden_states = input_encoder_hidden_states.clone()
                        block_encoder_hidden_states[history_indices] = spatial_history_enc
                elif layer in temporal_layers and temporal_history_states is not None:
                    block_hidden_states = hidden_states.clone()
                    block_hidden_states[history_indices] = temporal_history_states
                    if block_encoder_hidden_states is not None and temporal_history_enc is not None:
                        block_encoder_hidden_states = input_encoder_hidden_states.clone()
                        block_encoder_hidden_states[history_indices] = temporal_history_enc

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    block_hidden_states,
                    block_encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    skip,
                    input_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    block_hidden_states,
                    encoder_hidden_states=block_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=input_attention_kwargs,
                )

            if history_indices is not None:
                if layer in spatial_layers and spatial_history_states is not None:
                    hidden_states[history_indices] = spatial_history_states
                elif layer in temporal_layers and temporal_history_states is not None:
                    hidden_states[history_indices] = temporal_history_states

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -num_tokens:]
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)


    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)


class PartCrafterDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    TripoSG: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
        use_style_cond_and_image_meta_size (`bool`,  *optional*):
            Whether or not to use style condition and image meta size. True for version <=1.1, False for version >= 1.2
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        max_num_parts: int = 32, 
        enable_part_embedding=True,
        enable_local_cross_attn: bool = True,
        enable_global_cross_attn: bool = True,
        global_attn_block_ids: Optional[List[int]] = None,
        global_attn_block_id_range: Optional[List[int]] = None,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )

        if enable_part_embedding:
            self.part_embedding = nn.Embedding(max_num_parts, self.inner_dim)
            self.part_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.enable_part_embedding = enable_part_embedding

        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_norm_type=None,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",  # TODO
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,  # this is an error
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    qkv_bias=False,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.enable_local_cross_attn = enable_local_cross_attn
        self.enable_global_cross_attn = enable_global_cross_attn

        if global_attn_block_ids is None:
            global_attn_block_ids = []
            if global_attn_block_id_range is not None:
                global_attn_block_ids = list(range(global_attn_block_id_range[0], global_attn_block_id_range[1] + 1))
        self.global_attn_block_ids = global_attn_block_ids

        if len(global_attn_block_ids) > 0:
            # Override self-attention processors for global attention blocks
            attn_processor_dict = {}
            modified_attn_processor = []
            for layer_id in range(num_layers):
                for attn_id in [1, 2]:
                    if layer_id in global_attn_block_ids:
                        # apply to both self-attention and cross-attention
                        attn_processor_dict[f'blocks.{layer_id}.attn{attn_id}.processor'] = PartCrafterAttnProcessor()
                        modified_attn_processor.append(f'blocks.{layer_id}.attn{attn_id}.processor')
                    else:
                        attn_processor_dict[f'blocks.{layer_id}.attn{attn_id}.processor'] = TripoSGAttnProcessor2_0()
            self.set_attn_processor(attn_processor_dict)
            # logger.info(f"Modified {modified_attn_processor} to PartCrafterAttnProcessor")

    def _set_gradient_checkpointing(
        self, 
        enable: bool = False, 
        gradient_checkpointing_func: Optional[Callable] = None,
    ):
        # TODO: implement gradient checkpointing
        self.gradient_checkpointing = enable

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        _, T, _ = hidden_states.shape

        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        hidden_states = self.proj_in(hidden_states)

        # T + 1 token
        hidden_states = torch.cat([temb, hidden_states], dim=1) # (N, T+1, D)

        if self.enable_part_embedding:
            # Add part embedding
            num_parts = attention_kwargs["num_parts"]
            if isinstance(num_parts, torch.Tensor):
                part_embeddings = []
                for num_part in num_parts:
                    part_embedding = self.part_embedding(torch.arange(num_part, device=hidden_states.device)) # (n, D)
                    part_embeddings.append(part_embedding)
                part_embedding = torch.cat(part_embeddings, dim=0) # (N, D)
            elif isinstance(num_parts, int):
                part_embedding = self.part_embedding(torch.arange(hidden_states.shape[0], device=hidden_states.device)) # (N, D)
            else:
                raise ValueError(
                    "num_parts must be a torch.Tensor or int, but got {}".format(type(num_parts))
                )
            hidden_states = hidden_states + part_embedding.unsqueeze(dim=1) # (N, T+1, D)

        # prepare negative encoder_hidden_states
        negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states) if encoder_hidden_states is not None else None

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()
            if (
                (not self.enable_local_cross_attn) 
                and len(self.global_attn_block_ids) > 0
                and (layer not in self.global_attn_block_ids)
            ):
                # If in non-global attention block and disable local cross attention, use negative encoder_hidden_states
                # Do not inject control signal into non-global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            elif (
                (not self.enable_global_cross_attn)
                and len(self.global_attn_block_ids) > 0
                and (layer in self.global_attn_block_ids)
            ):
                # If in global attention block and disable global cross attention, use negative encoder_hidden_states
                # Do not inject control signal into global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            else:
                input_encoder_hidden_states = encoder_hidden_states
            
            if len(self.global_attn_block_ids) > 0 and (layer in self.global_attn_block_ids):
                # Inject control signal into global attention block
                input_attention_kwargs = attention_kwargs
            else:
                input_attention_kwargs = None

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    input_encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    skip,
                    input_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=input_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=input_attention_kwargs,
                )  # (N, T+1, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -T:]  # (N, T, D)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)


class TripoSGDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    TripoSG: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
        use_style_cond_and_image_meta_size (`bool`,  *optional*):
            Whether or not to use style condition and image meta size. True for version <=1.1, False for version >= 1.2
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        use_cross_attention_2: bool = False,
        cross_attention_2_dim: Optional[int] = None
    ):
        super().__init__()
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )
        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_norm_type=None,
                    use_cross_attention_2=use_cross_attention_2,
                    cross_attention_2_dim=cross_attention_2_dim,
                    cross_attention_2_norm_type=None,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",  # TODO
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,  # this is an error
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    qkv_bias=False,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        _, N, _ = hidden_states.shape

        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        hidden_states = self.proj_in(hidden_states)

        # N + 1 token
        hidden_states = torch.cat([temb, hidden_states], dim=1)

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_2,
                    temb,
                    image_rotary_emb,
                    skip,
                    attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                )  # (N, L, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -N:]
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)
