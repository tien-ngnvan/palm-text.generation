# cerebro-GenPALM
```bash
cd cerebro-GenPALM
```
## Installation 
- [pytorch](https://pytorch.org/)

## Download example dataset (CNN-DM Data)
- [CNN Stories](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ)
- [DailyMail Stories](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs)
- After downloading these two dataset, unzip them, and there will be two folder ```cnn``` and ```dailymail``` 
## Run Code 
### Make datafiles 
```bash
python -m utils.make_datafiles ./cnn/stories ./daiymail/stories 
```
- This step will get data and split the dataset into train/val/test 
- Each train/ val/ test contains two files: *.source and *.target 
    - *.source file is the input for the encoder model 
    - *.target file is the input for the decoder model and also used as label to compute loss. 

### BPE preprocess 
- If there is no ```encoder.json```, ```vocab.bpe``` and ```dict.txt``` existing, download them and execute bpe preprocessing step
```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

sh scripts/bpe_preprocess.sh
```
- It is similar to tokenizing step that transforms text format to ids

### Binarize dataset 
- The previous step converts texts to ids. However, all ids are in string data type. This step will map them to tensor (pytorch tensor) for computation
```bash
sh scripts/palm-preprocess.sh
```

### Train model without pretraining
```bash
sh scripts/palm-train.sh
```

## Pretrain PALM
```bash
cd cerebro-GenPALM
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
mkdir corpus
mkdir corpus/wiki
mkdir corpus/wikitext
mv enwiki-latest-pages-articles.xml.bz2 corpus
python3 -m utils.process_wiki
```
