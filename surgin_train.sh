#!/bin/bash
task=wound
prompt=prompt_manual

config_file=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
model_checkpoint=MODEL/GLIP/glip_tiny_model_o365_goldg_cc_sbu.pth
prompt_config=configs/odinm_X/${task}/prompt/${prompt}.yaml




task_configs=("1.yaml" "2.yaml" "3.yaml" "4.yaml" "5.yaml")
names=("t1" "t2" "t3" "t4" "t5")
custom_shot_and_epoch_and_general_copys=("0_50_1" "0_50_1" "0_50_1" "0_50_1" "0_50_1")

NumberOfRuns=${#task_configs[@]}
for ((i=0; i<$NumberOfRuns; i++))
  do
      echo ${task_configs[i]} ${names[i]} ${custom_shot_and_epoch_and_general_copys[i]}
      task_config=${task_configs[i]}
      name=${names[i]}
      custom_shot_and_epoch_and_general_copy=${custom_shot_and_epoch_and_general_copys[i]}
      echo ${config_file}
      output_dir=OUTPUT1/GLIP-T/${name}

      CUDA_VISIBLE_DEVICES=0 python tools/fine_tuning.py  \
          --config-file ${config_file}    --ft-tasks configs/odinm_X/${task}/${task_config}  --skip-test \
          --custom_shot_and_epoch_and_general_copy ${custom_shot_and_epoch_and_general_copy} \
          --evaluate_only_best_on_test --push_both_val_and_test \
          OUTPUT_DIR ${output_dir} \
          MODEL.WEIGHT ${model_checkpoint} \
          SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 SOLVER.WEIGHT_DECAY 0.05  SOLVER.BASE_LR 0.0001    SOLVER.WARMUP_FACTOR 0.01  SOLVER.LANG_LR 0.00001\
          TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True\
          MODEL.BACKBONE.USE_CHECKPOINT True   MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False \
          SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
          SOLVER.STEP_PATIENCE 3 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE  full
  done
