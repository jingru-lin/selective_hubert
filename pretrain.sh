# Pretraining:

###########################################On Simulted Data#############################################################
CUDA_VISIBLE_DEVICES=4,5,6 fairseq-hydra-train \
  --config-dir /dir/to/selective_hubert/config \
  --config-name hubert_contrastive_ref2_base_librispeech.yaml \
  task.data=/dir/to/data \
  task.label_dir=/dir/to/label \
  task.labels='["km"]' \
  model.label_rate=50 \
  distributed_training.distributed_world_size=3 \
  distributed_training.nprocs_per_node=8 \
  checkpoint.save_dir=/save/dir \
  common.user_dir=/dir/to/selective_hubert \
  model.pretrained_ckpt_path=/dir/to/pretrained/hubert_base_ls960.pt \
  model.speaker_injection_layers=[0] \
  criterion.loss_weights=[10,1] \
  optimization.update_freq=[4] \
  optimization.lr=[0.000075] \
  optimization.max_update=400000 \
  dataset.num_workers=12 \
  dataset.train_subset=train_960h \
  dataset.valid_subset=dev-other \
  dataset.max_tokens=3800000 \
  task.max_sample_size=250000 \
  task.spk_embedding_dir=/dir/to/extracted/cam_embeddings \
  task.noise_apply_prob=0.3 \
  task.get_multiple_prob=0.95 \
  task.contrastive_data=True \
  model.ctr_layer=-1 \
  model.speaker_dim=192
