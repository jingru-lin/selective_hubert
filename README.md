# Selective-HuBERT

### Step 0: Clone repo
```
git clone -b pretrain https://github.com/jingru-lin/selective_hubert.git
```
### Step 1: 
Follow https://github.com/facebookresearch/fairseq/tree/main and install fairseq (version 0.12.0)

### Step 2: Prepare tsv and km label file
Follow fairseq hubert documentation to produce hubert tsv files and kmeans labels: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

### Step 3: Get speaker embeddings
Follow https://github.com/alibaba-damo-academy/3D-Speaker to extract speaker embeddings
Use model_id=iic/speech_campplus_sv_zh-cn_16k-common

### Step 4: Download pre-trained hubert
From https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

### Step 5: Start pretraining
To pretrain the model, use the provided config that contains the default hyperparameters.
Change the following paths in pretrain.sh and run it:
```
--config-dir /dir/to/selective_hubert/config
--config-name hubert_contrastive_ref2_base_librispeech.yaml
task.data=/dir/to/data
task.label_dir=/dir/to/label
common.user_dir=/dir/to/selective_hubert
model.pretrained_ckpt_path=/dir/to/pretrained/hubert_base_ls960.pt
task.spk_embedding_dir=/dir/to/extracted/cam_embeddings
```
