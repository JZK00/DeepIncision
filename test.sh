#!/bin/bash
GPU=$1
task=wound
prompt=prompt_manual

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


task_configs=("1.yaml" "2.yaml" "3.yaml" "4.yaml" "5.yaml")
names=("1" "2" "3" "4" "5")
custom_shot_and_epoch_and_general_copys=("0_50_1" "0_50_1" "0_50_1" "0_50_1" "0_50_1")



NumberOfRuns=${#task_configs[@]}
# NumberOfRuns=1
for ((i=0; i<$NumberOfRuns; i++))
  do
      echo ${task_configs[i]} ${names[i]} ${custom_shot_and_epoch_and_general_copys[i]}
      task_config=${task_configs[i]}
      name=${names[i]}
      custom_shot_and_epoch_and_general_copy=${custom_shot_and_epoch_and_general_copys[i]}
      echo ${config_file}
      config_file=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
      prompt_config=configs/odinm_X/wound/1.yaml
      model_checkpoint="model_path/ft_${name}_task1/model_best.pth"

      output_dir=OUTPUT/${task}/ft_${name}
      CUDA_VISIBLE_DEVICES=${GPU} python tools/test_grounding_net.py --config-file ${config_file} --weight ${model_checkpoint} \
        --prompt-file ${prompt_config} \
        TEST.IMS_PER_BATCH 2 \
        DATALOADER.NUM_WORKERS 4 \
        MODEL.DYHEAD.SCORE_AGG "MEAN" \
        INPUT.MIN_SIZE_TEST ${IMG_S} \
        TEST.EVAL_TASK detection \
        MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
        OUTPUT_DIR ${output_dir}
  done
