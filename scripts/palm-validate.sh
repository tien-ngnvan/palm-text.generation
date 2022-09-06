#DATA_BIN=/datadrive/cnn_dm_bin
#task=wikitext
task=cnn_dm

CUDA_VISIBLE_DEVICES=0 python /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/utils/validate.py /content/drive/MyDrive/Modified-models/Implement_PALM/corpus/"$task"_bin \
--user-dir /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/src --truncate-source --source-lang source --target-lang target \
--task auto_encoding_regressive --criterion label_smoothed_cross_entropy_with_masked_lm \
--skip-invalid-size-inputs-valid-test \
--max-tokens 4096 --required-batch-size-multiple 1 \
--valid-subset valid \
--max-tokens-valid 4096 \
--path /content/drive/MyDrive/test_tensorboard/palm/palm_"${task}"_checkpoints/checkpoint_best.pt

# CUDA_VISIBLE_DEVICES=0 fairseq-validate $DATA_BIN \
# --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.0001 --act-dropout 0.1 \
# --lr-scheduler polynomial_decay --total-num-update 20000 --warmup-updates 500 --label-smoothing 0.1 \
# --eval-bleu-print-samples \
# --validate-interval 1 \
# --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu \

 
