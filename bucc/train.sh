#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4 fairseq-train data/UNv1.0.bpe40k-bin --max-epoch 4 --ddp-backend=no_c10d --task multilingual_translation_singlemodel --lang-pairs en-es,es-en,fr-en,ru-en,zh-en --arch multilingual_lstm_laser_mseLearn --optimizer adam --adam-betas '(0.9, 0.98)'  --lr 0.001 --min-lr '1e-09' --label-smoothing 0.1 --criterion label_smoothed_cross_entropy_with_langid --dropout 0.1 --user-dir ../fairseq-modules/ --max-tokens 8000  --update-freq 1 --memory-efficient-fp16 --save-dir checkpoints/UNv1.0 --lr-scheduler inverse_sqrt --min-loss-scale '1e-09' --warmup-updates 4000 --warmup-init-lr 0.001

