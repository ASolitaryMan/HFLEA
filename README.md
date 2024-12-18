# Frame-level emotional state alignment method for speech emotion recognition 

Official implemention for the paper "Frame-level emotion state alignment method for speech emotion recognition", which is submitted to ICASSP 2024.

## I am very sorry, I have carefully checked the data division method. The method we used is leave-one-session-out https://github.com/ECNU-Cross-Innovation-Lab/ShiftSER. For each fold, four sessions are used for training while one session is used for testing. The final result is the average of 5 folds. I made a mistake in my paper, and I appreciate everyone's correction. I am extremely sorry for the trouble caused to the readers.

## Libraries and dependencies

1. pytorch
2. PyTorch-lighting
3. accelerater
4. sklearn
5. transformers
6. soudfile

## Framework of method

![image](https://github.com/ASolitaryMan/HFLEA/assets/42258022/dca6f20b-a4ca-4edd-b616-5248bdbb6194)

## Results on IEMOCAP

![image](https://github.com/ASolitaryMan/HFLEA/assets/42258022/08ea33f5-016a-4938-be7a-c024e73cb782)

## The format of training, validation, and test sources

You need  three scp files containing training, validation and test sources, which are divided according to the methods described in the paper . These scp files are shown as following format (**wav_name, wav_path, wav_label)**:

![image](https://github.com/ASolitaryMan/HFLEA/assets/42258022/0f7e245b-0087-464c-91fe-2a00ebd18bdf)

## Phase 1. TPAT and Cluster

### TAPT

#### Task adaptive pretraining stage

First, we need to dump feature and generate pesudo label for pretraining by a pretrained k-means model which is provided by official HuBERT https://github.com/facebookresearch/fairseq/tree/main/examples/hubert. Second, you need copy 5531 samples in IEMOCAP into a folder, for example "./audio". You can find the "hubert-base-ls960" at here https://huggingface.co/facebook/hubert-base-ls960.

```
#dump feature and generate pesudo label
python preprocess.py --root_dir ./audio 
                     --feat_type hubert 
                     --exp_dir ./exp 
                     --layer_index 9 
                     --num_rank 40 
                     --hubert_base ./hubert-base-ls960 
                     --checkpoint_path ./hubert-base-ls960 
                     --iskmean False
                     --pretrain_kmeans hubert_base_ls960_L9_km500.bin 
                     --percent -1
```

```
#pretraining hubert
python train.py --dataset_path ./exp/data/hubert_9/ 
                --hubert_path ./hubert-base-ls960 
                --exp_dir ./exp_iter2_9_500 
                --feature_type hubert 
                --num_class 500 
                --max_updates 20000 
                --learning_rate 0.0005 
                --gpus 4 
                --num_nodes 1
```

#### Task adaptive fine-tuning stage

```
#transfer the format of model into the format of huggingface.
python model_transfer.py --pretrained_model_path ./exp_iter2_9_500/epoch=179-step=17250.ckpt               											    --hubert_base_ls960 ./hubert-base-ls960 
                         --save_path ./transfer_model_9_500
```

```
#fine-tune hubert
accelerate fine_tune_hubert.py --CPT_HuBERT_path ./transfer_model_9_500 
                               --train_src_path ./session01_train.scp 
                               --valid_src_path ./session01_valid.scp 
                               --test_src_path ./session01_test.scp 
                               --save_root ./session01_fine_tune_model
```

### Cluster

**From here on down, it's all about session 1. You need to repeat the following steps on other sessions.**

```
python preprocess.py --root_dir ./audio 
                     --feat_type hubert 
                     --exp_dir ./exp 
                     --layer_index 9 
                     --num_rank 40 
                     --hubert_base ./hubert-base-ls960 
                     --checkpoint_path ./session01_fine_tune_model 
                     --num_cluster 50 
                     --percent -1
```

## Phase 2. Pretraining HuBERT

```
python train.py --dataset_path ./exp/data/hubert_9/ 
                --hubert_path ./hubert-base-ls960 
                --exp_dir ./exp_iter2_9_50_session01 
                --feature_type hubert 
                --num_class 50 
                --max_updates 20000 
                --learning_rate 0.0005 
                --gpus 4 
                --num_nodes 1
```

```
python model_transfer.py --pretrained_model_path ./exp_iter2_9_50_session01/epoch=168-step=16196.ckpt 
                         --hubert_base_ls960 ./hubert-base-ls960 
                         --save_path ./transfer_model_9_50_session01_CPT_HuBERT
```

## Phase 3. Fine-tuning CPT-HuBERT for SER

```
accelerate fine_tune_hubert.py --CPT_HuBERT_path ./transfer_model_9_50_session01_CPT_HuBERT           															 
                               --train_src_path ./session01_train.scp 
                               --valid_src_path ./session01_valid.scp 
                               --test_src_path ./session01_test.scp 
                               --save_root 	./session01_final_model_SER
```

