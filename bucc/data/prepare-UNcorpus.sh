#!/bin/bash

SRCS=(
    "es"
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

DATA=$ROOT/UNv1

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

#Tokenizing Chinese dev set with the library Jieba
#cat $DATA/en-zh/UNv1.0.en-zh.zh | python -c 'import jieba
#import sys
#for line in sys.stdin:
#    print(" ".join(jieba.cut(line.strip(), cut_all=True)))' > $DATA/en-zh/UNv1.0.en-zh.lower.tok.zh

#cat $DATA/en-zh/UNv1.0.en-zh.en | $ROOT/../../preprocess/moses/tokenizer/tokenizer.perl -l en | awk '{ print tolower($0) }' > $DATA/en-zh/UNv1.0.en-zh.lower.tok.en 

#Tokenizing and lowercasing Spanish, French, Russian and English devsets
#for LANG in "es" "fr" "ru"; do
#	cat $DATA/en-$LANG/UNv1.0.en-$LANG.$LANG | $ROOT/../../preprocess/moses/tokenizer/tokenizer.perl -l $LANG | awk '{ print tolower($0) }' > $DATA/en-$LANG/UNv1.0.en-$LANG.lower.tok.$LANG &
#	cat $DATA/en-$LANG/UNv1.0.en-$LANG.en    | $ROOT/../../preprocess/moses/tokenizer/tokenizer.perl -l en    | awk '{ print tolower($0) }' > $DATA/en-$LANG/UNv1.0.en-$LANG.lower.tok.en &
#done
#
#wait

# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/en-$SRC/UNv1.0.en-$SRC.lower.tok.$SRC; echo $DATA/en-$SRC/UNv1.0.en-$SRC.lower.tok.en; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe \
    --shuffle_input_sentence=true


# encode train/valid/test
echo "encoding train/valid with learned BPE..."
for SRC in "${SRCS[@]}"; do
    for LANG in "$SRC" "$TGT"; do
        python "$SPM_ENCODE" \
            --model "$DATA/sentencepiece.bpe.model" \
            --output_format=piece \
            --inputs $DATA/UNv1.0.en-$SRC.lower.tok.${SRC} $DATA/UNv1.0.en-$SRC.lower.tok.${TGT} \
            --outputs $DATA/UNv1.0.en-$SRC.bpe.${SRC} $DATA/UNv1.0.en-$SRC.bpe.${TGT} \
            --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN 
    done
done

DATABIN=$ROOT/UNv1.0.bpe40k-bin

fairseq-preprocess --source-lang en --target-lang fr --trainpref $DATA/UNv1.0.trainset.bpe.en-fr --validpref $DATA/UNv1.0.devset.bpe --joined-dictionary --destdir $DATABIN --workers 15 --srcdict $DATABIN/dict.en.txt
fairseq-preprocess --source-lang en --target-lang ru --trainpref $DATA/UNv1.0.trainset.bpe.en-ru --validpref $DATA/UNv1.0.devset.bpe --joined-dictionary --destdir $DATABIN --workers 15 --srcdict $DATABIN/dict.en.txt
fairseq-preprocess --source-lang en --target-lang zh --trainpref $DATA/UNv1.0.trainset.bpe.en-zh --validpref $DATA/UNv1.0.devset.bpe --joined-dictionary --destdir $DATABIN --workers 15 --srcdict $DATABIN/dict.en.txt

