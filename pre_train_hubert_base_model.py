from transformers import HubertConfig
import warnings
from typing import Optional, Tuple, Union
from transformers import HubertPreTrainedModel

from torch import Tensor
from torch.nn import Parameter
import torch
from torch import nn
import numpy as np
from transformers.utils.generic import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass
from transformers.models.hubert.modeling_hubert import HubertFeatureEncoder, HubertFeatureProjection, HubertEncoderStableLayerNorm, HubertEncoder
from transformers import AutoConfig, HubertModel

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

@dataclass
class PreTrainBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    features_pen: Optional[Tuple[torch.FloatTensor]] = None
    logit_m: torch.FloatTensor = None
    logit_u: torch.FloatTensor = None
    features_pen: int = None

# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool_)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask

def _compute_logits(
    proj_x: Tensor,
    target: Tensor,
    label_embeddings: Parameter,
) -> Tensor:
    """Compute the logits of the embeddings.
    Args:
        proj_x (Tensor): The projected masked representations of dimension `[batch, frame, final_dim]`.
        target (Tensor): The target Tensor of dimension `[batch, frame, final_dim]`.
        label_embeddings (Parameter): The trainable embeddings of target of dimension `[num_class, final_dim]`.

    Returns:
        (Tensor): The logits of the inputs.
    """
    logit_temp = 0.1
    pos = torch.index_select(label_embeddings, 0, target.long())
    negs = label_embeddings.unsqueeze(1).expand(-1, proj_x.size(0), -1)
    neg_is_pos = (pos == negs).all(-1)
    pos = pos.unsqueeze(0)
    targets = torch.cat([pos, negs], dim=0)

    logits = torch.cosine_similarity(proj_x.float(), targets.float(), dim=-1).type_as(proj_x)
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")
    logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    return logits

class LogitGenerator(nn.Module):
    """Generate the logits of masked and unmasked inputs.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        num_classes (int): The number of classes in the labels.
        final_dim (int): Project final representations and targets to `final_dim`.
        skip_masked (bool): If True, skip computing losses over masked frames.
        skip_nomask (bool): If True, skip computing losses over unmasked frames.
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        skip_masked: bool =False,
        skip_nomask: bool =False,
    ):
        super().__init__()
        self.label_embeddings = Parameter(torch.FloatTensor(num_classes, final_dim))
        torch.nn.init.uniform_(self.label_embeddings)
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask

    def forward(self, x: Tensor, label: Tensor, mask_m: Tensor, mask_u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): The feature representation of the last transformer layer.
            label (Tensor): The label Tensor of dimension `[batch, frame]`.
            mask_m (Tensor): The masked indices of dimension `[batch, frame]`.
            mask_u (Tensor): The unmasked indices of dimension `[batch, frame]`.

        Returns:
            Tensor: The logits of masked frames. Tensor of dimension `[masked_frame, final_dim]`.
            Tensor: The logits of unmasked frames. Tensor of dimension `[unmasked_frame, final_dim]`.
        """
        proj_x = self.final_proj(x)
        if self.skip_masked:
            logit_m = None
        else:
            proj_x_m = proj_x[:][mask_m]
            label_m = label[:][mask_m]
            logit_m = _compute_logits(proj_x_m, label_m, self.label_embeddings)

        if self.skip_nomask:
            logit_u = None
        else:
            proj_x_u = proj_x[mask_u]
            label_u = label[mask_u]
            logit_u = _compute_logits(proj_x_u, label_u, self.label_embeddings)
        return logit_m, logit_u

class HubertModelPreTrainBase(HubertPreTrainedModel):
    def __init__(self, config: HubertConfig, num_cluster:int, feature_grad_mult:float):
        super().__init__(config)
        self.config = config
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)
        self.num_class = num_cluster

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            self.encoder = HubertEncoder(config)
        self.logit_generator = LogitGenerator(
            encoder_embed_dim=self.config.hidden_size,
            num_classes=self.num_class,
            final_dim=256,
            skip_masked=False,
            skip_nomask=False,
        )

        self.feature_grad_mult = feature_grad_mult

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            # if self.training:
            #     mask_time_indices = mask_time_indices
            # else:
            #     mask_time_indices[:] = False 
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states, mask_time_indices
    
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: torch.Tensor = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PreTrainBaseModelOutput]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import Wav2Vec2Processor, HubertModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        #GradMultiply
        if self.feature_grad_mult is not None and self.feature_grad_mult < 1.0 and self.training:
            extract_features = GradMultiply.apply(extract_features, self.feature_grad_mult)

        features_pen = extract_features.float().pow(2).mean()

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states, mask = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices) #x, mask

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        attention_mask = ~attention_mask
        if attention_mask is not None:
            mask_m = torch.logical_and(~attention_mask, mask)
            mask_u = torch.logical_and(~attention_mask, ~mask_m)
        else:
            mask_m = mask
            mask_u = ~mask_m

        logit_m, logit_u = self.logit_generator(hidden_states, labels, mask_m, mask_u)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return PreTrainBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            logit_m=logit_m,
            logit_u=logit_u,
            features_pen=features_pen
        )
    
    def inference(self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states, mask = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices) #x, mask

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class PreTrainHuBERTBase(HubertPreTrainedModel):
    def __init__(self, config, num_class=504, feature_grad_mult=0.1):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.hubert.encoder.gradient_checkpointing = False
        self.num_class = num_class
        self.feature_grad_mult = feature_grad_mult
        self.logit_generator = LogitGenerator(
            encoder_embed_dim=768,
            num_classes=self.num_class,
            final_dim=256,
            skip_masked=False,
            skip_nomask=False,
        )
    
    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.hubert.parameters():
            param.requires_grad = False

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.hubert.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            # if self.training:
            #     mask_time_indices = mask_time_indices
            # else:
            #     mask_time_indices[:] = False 
            hidden_states[mask_time_indices] = self.hubert.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states, mask_time_indices

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: torch.Tensor = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.hubert.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        #GradMultiply
        if self.feature_grad_mult is not None and self.feature_grad_mult < 1.0 and self.training:
            extract_features = GradMultiply.apply(extract_features, self.feature_grad_mult)

        features_pen = extract_features.float().pow(2).mean()

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.hubert.feature_projection(extract_features)
        hidden_states, mask = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices) #x, mask

        encoder_outputs = self.hubert.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        attention_mask = ~attention_mask
        if attention_mask is not None:
            mask_m = torch.logical_and(~attention_mask, mask)
            mask_u = torch.logical_and(~attention_mask, ~mask_m)
        else:
            mask_m = mask
            mask_u = ~mask_m

        logit_m, logit_u = self.logit_generator(hidden_states, labels, mask_m, mask_u)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return PreTrainBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            logit_m=logit_m,
            logit_u=logit_u,
            features_pen=features_pen
        )
    
    def inference(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

if __name__ == "__main__":
    pass
    # from transformers import AutoConfig
    # config = AutoConfig.from_pretrained("/home/lqf/workspace/icassp2023/hubert-base-ls960")
    # print(config.mask_time_prob)
    # model = HubertModelPreTrainBase(config, num_cluster=50).from_pretrained("/home/lqf/workspace/icassp2023/session05", num_cluster=50)
    # print(model.config.mask_time_prob)
    # print(model.num_class)
    
