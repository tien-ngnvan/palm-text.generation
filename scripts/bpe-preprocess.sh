<<<<<<< HEAD
for split in train val; do for lang in source target; do python utils/multiprocessing_bpe_encoder.py --encoder-json gpt2_bpe/encoder.json  --vocab-bpe gpt2_bpe/vocab.bpe --inputs "./cnn_dm/$split.$lang" --outputs "./cnn_dm/$split.bpe.$lang" --workers 60 --keep-empty; done; done
=======
for split in train val; do for lang in source target; do python utils/multiprocessing_bpe_encoder.py --encoder-json gpt2_bpe/encoder.json  --vocab-bpe gpt2_bpe/vocab.bpe --inputs "./cnn_dm/$split.$lang" --outputs "./cnn_dm/$split.bpe.$lang" --workers 60 --keep-empty; done; done
>>>>>>> 121c2e0babcf3424ba7a63f6afc4a8fa6a097790
