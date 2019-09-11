#!/bin/bash

SRCS=(
    "fr"
    "ru"
    "zh"
)
TGT="en"

ROOT=$(dirname "$0")

FAIRSEQ=$1
SCRIPTS=$FAIRSEQ/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=40000

DATA=$ROOT/bucc2018
MODEL=UNv1.0/sentencepiece.bpe.model

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

#Tokenizing Chinese test and training set with the library Jieba

cat $DATA/zh-en/zh-en.test.zh | python -c 'import jieba
import sys
for line in sys.stdin:
    print(" ".join(jieba.cut(line.strip(), cut_all=True)))' > $DATA/zh-en/zh-en.test.tok.zh

cat $DATA/zh-en/zh-en.test.en | preprocess/moses/tokenizer/tokenizer.perl -l en | awk '{ print tolower($0) }' > $DATA/zh-en/zh-en.test.tok.en 

#Tokenizing and lowercasing French and Russian training and test sets
for LANG in "fr" "ru"; do
	cut -f 2 $DATA/$LANG-en/$LANG-en.test.en | preprocess/moses/tokenizer/tokenizer.perl -l en | awk '{ print tolower($0) }' > $DATA/$LANG-en/$LANG-en.test.tok.en &
	cut -f 2 $DATA/$LANG-en/$LANG-en.test.$LANG | preprocess/moses/tokenizer/tokenizer.perl -l $LANG | awk '{ print tolower($0) }' > $DATA/$LANG-en/$LANG-en.test.tok.$LANG &
done

wait

# encode train/valid/test
echo "encoding train/test with learned BPE..."
for SRC in "${SRCS[@]}"; do
    python "$SPM_ENCODE" \
        --model "$MODEL" \
        --output_format=piece \
        --inputs $DATA/$SRC-en/$SRC-en.test.tok.$SRC $DATA/$SRC-en/$SRC-en.test.tok.en \
        --outputs $DATA/$SRC-en/$SRC-en.test.bpe.$SRC $DATA/$SRC-en/$SRC-en.test.bpe.en \
        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN 
done
