#TASK=wikitext
TASK=medical

for SPLIT in train val test
do
  for LANG in source target
  do
    python /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/utils/multiprocessing_bpe_encoder.py \
    --encoder-json /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/gpt2_bpe/encoder.json \
    --vocab-bpe /content/drive/MyDrive/Modified-models/Implement_PALM/PALM/gpt2_bpe/vocab.bpe \
    --inputs "/content/drive/MyDrive/Modified-models/Implement_PALM/Medical-dataset/output/$SPLIT.$LANG" \
    --outputs "/content/bpe/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
