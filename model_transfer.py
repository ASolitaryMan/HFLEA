import torch
from pre_train_hubert_base_model import HubertModelPreTrainBase
from transformers import AutoConfig
import argparse

def transfer_model(pretrain_path, save_path, hubert_base):
    config = AutoConfig.from_pretrained(hubert_base)
    model = HubertModelPreTrainBase(config)
    state_dict = torch.load(pretrain_path, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./transfer_model_11_50_s1', help='transfer the pretrained model to CPT-HuBERT')
    parser.add_argument('--pretrained_model_path', type=str, default='./epoch=179-step=17250.ckpt', help='The path of model which is pretrained with frame-level pesudo-emotion label')
    parser.add_argument('--hubert_base_ls960', type=str, default='./hubert-base-ls960', help='The path of hubert-base-ls960')
    
    args = parser.parse_args()

    pretrain_path = args.pretrained_model_path
    save_path = args.save_path
    hubert_base = args.hubert_base_ls960
    transfer_model(pretrain_path, save_path, hubert_base)