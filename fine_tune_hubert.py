import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    Wav2Vec2Model,
    HubertPreTrainedModel,
    AutoConfig,
)
import random
from sklearn.utils.class_weight import compute_class_weight
import soundfile as sf
import torch.nn as nn
from accelerate import Accelerator
import argparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
import math
from torch.utils.data.sampler import Sampler
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
# from ignite.distributed import DistributedProxySampler
from torch.utils.data import DataLoader
# from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from sklearn.metrics import confusion_matrix
from collections import Counter
import torch.nn.init as init
from lightning_fabric.utilities.seed import seed_everything
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention


emos = ["hap", 'neu', 'sad', 'ang']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

def read_text(text_path):
    f = open(text_path, "r")
    lines = f.readlines()
    f.close()
    return lines

class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atten_weight = nn.Parameter(torch.Tensor(hidden_dim, 1), requires_grad=True)
        # self.bais_weight = nn.Parameter(torch.zeros(time_setps), requires_grad=True)
        nn.init.uniform_(self.atten_weight)

    def compute_mask(self, inputs, mask):
        # mask = mask.unsqueeze(0)
        new_attn_mask = torch.zeros_like(mask, dtype=inputs.dtype)
        new_attn_mask.masked_fill_(mask, float("-inf")) #mask是True

        return new_attn_mask

    def forward(self, inputs, mask=None):
        
        eij = torch.matmul(inputs, self.atten_weight).squeeze(-1)
        
        eij = torch.tanh(eij)

        if mask is not None:
            mask = ~mask
            tmask = self.compute_mask(inputs, mask)
            # print(tmask.size())
            a = torch.softmax(eij+tmask, dim=1).unsqueeze(-1)
        else:
            a = torch.softmax(eij, dim=1).unsqueeze(-1)

        weighted_output = inputs * a

        return weighted_output.sum(dim=1)
    
class AttentiveStatsPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atten_weight = nn.Parameter(torch.Tensor(hidden_dim, 1), requires_grad=True)
        # self.bais_weight = nn.Parameter(torch.zeros(time_setps), requires_grad=True)
        nn.init.uniform_(self.atten_weight)

    def compute_mask(self, inputs, mask):
        # mask = mask.unsqueeze(0)
        new_attn_mask = torch.zeros_like(mask, dtype=inputs.dtype)
        new_attn_mask.masked_fill_(mask, float("-inf")) #mask是True

        return new_attn_mask

    def forward(self, inputs, mask=None):
        
        eij = torch.matmul(inputs, self.atten_weight).squeeze(-1)
        
        eij = torch.relu(eij)

        if mask is not None:
            mask = ~mask
            tmask = self.compute_mask(inputs, mask)
            # print(tmask)
            a = torch.softmax(eij+tmask, dim=1).unsqueeze(-1)
        else:
            a = torch.softmax(eij, dim=1).unsqueeze(-1)

        weighted_output = inputs * a

        noise = 1e-5*torch.randn(weighted_output.size())

        if inputs.is_cuda:
            noise = noise.to(inputs.device)
        avg_repr, std_repr = weighted_output.sum(1), (weighted_output+noise).std(1)

        representations = torch.cat((avg_repr,std_repr),1)

        return representations

class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)


        return attn_output.sum(dim=1).squeeze(1)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_features, emo_label = [], []
        sample_rate = self.feature_extractor.sampling_rate

        for feature in features:
            input_features.append({"input_values": feature[0]})
            emo_label.append(feature[1])

        
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length * sample_rate,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=True,
            return_tensors="pt",
        )

        d_type = torch.long if isinstance(emo_label[0], int) else torch.float32
        batch["emo_labels"] = torch.tensor(emo_label, dtype=d_type)

        return batch


class DistributedWeightedSampler(Sampler):
    """
    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same across
        all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
        self,
        weights: list,
        num_samples: int = None,
        replacement: bool = True,
        num_replicas: int = None,
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = num_replicas
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """
        rank = os.environ["LOCAL_RANK"]

        rank = int(rank)

        if rank >= self.num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(rank, self.num_replicas - 1)
            )

        # weights = self.weights.copy()
        weights = deepcopy(self.weights)
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[rank : self.total_num_samples : self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[rank : self.total_num_samples : self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica

class MERDataset(Dataset):

    def __init__(self, src_path):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []
        for line in all_lines:
            tmp = line.strip().split("\n")[0].split()
            self.wav_list.append(tmp[1])
            self.label.append(emo2idx[tmp[-1]])
        
    def __getitem__(self, index):

        wave, sr = sf.read(self.wav_list[index])
        assert sr == 16000
        lab = self.label[index]
    
        return torch.FloatTensor(wave), lab

    def __len__(self):
        return len(self.label)
    
    def class_weight_v(self):
        labels = np.array(self.label)
        class_weight = torch.tensor([1/x for x in np.bincount(labels)], dtype=torch.float32)
        return class_weight
    
    def class_weight_q(self):
        class_weight = self.class_weight_v()
        return class_weight / class_weight.sum()
    
    def class_weight_k(self):
        labels = np.array(self.label)
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        weight = weight.tolist()
        samples_weight = torch.tensor([weight[t] for t in labels], dtype=torch.float32)
        """
        class_sample_count = np.unique(labels, return_counts=True)[1]
        class_sample_count = class_sample_count / len(label)
        weight = 1 / class_sample_count
        """
        return samples_weight
    
    def class_weight(self):
        self.emos = Counter(self.label)
        self.emoset = [0,1,2,3]
        weights = torch.tensor([self.emos[c] for c in self.emoset]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()

        return weights

def get_loaders(args, train_path, valid_path, test_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/lqf/workspace/icassp2023/hubert-base-ls960", return_attention_mask=True)
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True, max_length=args.max_length)

    train_dataset = MERDataset(train_path)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MERDataset(valid_path)
    test_dataset = MERDataset(test_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())
    # sampler = DistributedProxySampler(
    # ExhaustiveWeightedRandomSampler(class_weight, num_samples=train_dataset.__len__()))
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)


    return train_dataloader, valid_dataloader, test_dataloader, class_weight

class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self):
        super().__init__()
        # self.dense1 = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        # self.dense2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        self.fc_out_1 = nn.Linear(768, 4)
        
    def forward(self, features):
        # x = self.dense1(features)
        # feat = self.dense2(x)

        emos_out  = self.fc_out_1(features)
       
        return emos_out
    
class HubertForClassification(HubertPreTrainedModel):
    def __init__(self, config, pooling_mode="mean"):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.hubert.encoder.gradient_checkpointing = False

        self.dropout = nn.Dropout(self.hubert.config.final_dropout)
        self.atten = SoftAttention(768)
        # self.atten = AttentiveStatsPool(768)
        # self.atten = SelfAttention(embed_dim=768, num_heads=1, dropout=0.1)

        self.pooling_mode = pooling_mode

        self.classifier = HubertClassificationHead()
    
    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.hubert.parameters():
            param.requires_grad = False

    def merged_strategy(self, hidden_states, mask, mode="mean"):
        if mode == "mean":
            outputs = hidden_states.sum(dim=1) / mask.sum(dim=1).view(-1, 1)
        elif mode == "atten":
            outputs = self.atten(hidden_states, mask)
            # outputs = self.atten(hidden_states=hidden_states, attention_mask=mask)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'atten']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.hubert(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.hubert._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)
        emos_out = self.classifier(hidden_states)

        return emos_out


########################################################
########### main training/testing function #############
########################################################
def unweightedacc(y_true, y_pred):
    ua = 0.0
    cm = confusion_matrix(y_true, y_pred)

    for i in range(len(cm)):
        tmp = cm[i]
        ua += (tmp[i] / np.sum(tmp))
    return (ua / len(cm))

def train_model(accelerator, model, cls_loss, dataloader, optimizer=None, train=False):
    
    emo_probs, emo_labels = [], []
    batch_losses = []

    assert not train or optimizer!=None
    
    model.train()

    for data in tqdm(dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]
        
        ## add cuda
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            emos_out = model(input_values=input_values, attention_mask=attention_mask)
            loss = cls_loss(emos_out, emos)
            accelerator.backward(loss)
            optimizer.step()
        batch_losses.append(loss.item())

        all_emos_out, all_emos = accelerator.gather_for_metrics((emos_out, emos))
        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua = unweightedacc(emo_labels, emo_preds)

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_ua'] = emo_ua
    save_results['emo_wa'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels
    save_results['train_loss'] = np.array(batch_losses).mean()
    # item3: latent embeddings
    return save_results

def eval_model(accelerator, model, cls_loss, dataloader, optimizer=None, train=False):
    
    emo_probs, emo_labels = [], []
    
    model.eval()

    for data in tqdm(dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]

        with accelerator.autocast():
            with torch.no_grad():
                emos_out = model(input_values=input_values, attention_mask=attention_mask)

        all_emos_out,  all_emos = accelerator.gather_for_metrics((emos_out, emos))

        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua = unweightedacc(emo_labels, emo_preds)

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_ua'] = emo_ua
    save_results['emo_wa'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels

    return save_results

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--save_root', type=str, default='./session01', help='save prediction results and models')
    parser.add_argument('--CPT_HuBERT_path', type=str, default='./tranfer_model_9_50_s1', help='The path of CPT-HuBERT')
    parser.add_argument('--train_src_path', type=str, default='./session1_train.scp', help='The path of train src')
    parser.add_argument('--valid_src_path', type=str, default='./session1_dev.scp', help='The path of valid src')
    parser.add_argument('--test_src_path', type=str, default='./session1_test.scp', help='The path of test src')

    ## Params for model
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes [defined by args.label_path]')
    parser.add_argument('--pooling_model', type=str, default="mean", help="method for aggregating frame-level into utterence-level")

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')#0.00001
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=1234, help='make split manner is same with same seed')
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, default=10, help='max length of audio')
 
    
    args = parser.parse_args()

    # setup_seed(seed=args.seed)
    seed_everything(args.seed)

    train_src_path = args.train_src_path
    valid_src_path = args.valid_src_path
    test_src_path = args.test_src_path
    model_path = args.save_root   
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    max_eval_metric = -100

    accelerator.print (f'====== Reading Data =======')
    # train_loaders, eval_loaders, test_loaders, adim, tdim, vdim = get_loaders(args, config) 
    train_loader, eval_loader, test_loader, class_weight = get_loaders(args, train_src_path, valid_src_path, test_src_path)  

    if accelerator.is_main_process:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    accelerator.print (f'====== Training and Evaluation =======')

    accelerator.print (f'Step1: build model (each folder has its own model)')

    model = HubertForClassification.from_pretrained(args.CPT_HuBERT_path, pooling_mode="atten")
    model.hubert.config.mask_time_prob=0.08
    model.freeze_feature_extractor()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    model, optimizer, train_loader, eval_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader, test_loader)
    device = accelerator.device
    class_weight = class_weight.to(device)
    cls_loss = torch.nn.CrossEntropyLoss()

    max_eval_metric = -100
    max_eval_wa = -100
    max_test_metric = -100
    max_test_wa = -100

    accelerator.print (f'Step2: training (multiple epoches)')

    eval_fscores = []

    for epoch in range(args.epochs):

        ## training and validation
        train_results = train_model(accelerator, model, cls_loss, train_loader, optimizer=optimizer, train=True)
        eval_results  = eval_model(accelerator, model, cls_loss, eval_loader,  optimizer=None,      train=False)
        
        if accelerator.is_main_process:
            print ('epoch:%d; loss:%.4f, train_ua:%.4f, train_wa:%.4f, val_ua:%.4f; val_wa:%.4f' %(epoch+1, train_results['train_loss'], train_results['emo_ua'], train_results['emo_wa'], eval_results['emo_ua'], eval_results['emo_wa']))

        if max_eval_metric < eval_results['emo_ua']:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            milestone = model_path + "/" + "best_model_" + str(epoch)
            unwrapped_model.save_pretrained(milestone, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            max_eval_metric = eval_results['emo_ua']
            max_eval_wa = eval_results['emo_wa']
            eval_results  = eval_model(accelerator, model, cls_loss, test_loader,  optimizer=None,      train=False)
            max_test_metric = eval_results['emo_ua']
            max_test_wa = eval_results['emo_wa']
            best_ua = (max_eval_metric + max_test_metric) / 2
            best_wa = (max_eval_wa + max_test_wa) / 2
    
    accelerator.print('UA and WA:%.4f, %.4f' %(best_ua, best_wa))
            




    






