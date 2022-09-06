TASK=medical
#TASK=wikitext | cnn_dm

rm -rf /content/${TASK}_bin

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref /content/drive/MyDrive/Modified-models/Implement_PALM/Medical-dataset/bpe/train.bpe \
  --validpref /content/drive/MyDrive/Modified-models/Implement_PALM/Medical-dataset/bpe/val.bpe \
  --destdir /content/${TASK}_bin/ \
  --workers 60 \
  --srcdict /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/gpt2_bpe/dict.txt \
  --joined-dictionary
