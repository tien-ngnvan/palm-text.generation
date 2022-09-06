#DATA_BIN=/datadrive/cnn_dm_bin
#task=wikitext
task=cnn_dm

CUDA_VISIBLE_DEVICES=0 python /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/utils/interactive.py /content/drive/MyDrive/Modified-models/Implement_PALM/corpus/"$task"_bin \
#--user-dir /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/src --truncate-source --source-lang source --target-lang target \
#--task auto_encoding_regressive \
#--bpe gpt2 --beam 4 --lenpen 0.6 --nbest 1 \
#--path /content/drive/MyDrive/test_tensorboard/palm/palm_"${task}"_checkpoints/checkpoint_best.pt \
#--input /content/src.txt
