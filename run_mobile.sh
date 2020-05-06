#!/usr/bin/env bash

  CUDA_VISIBLE_DEVICES=0 python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=True \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=testdata   \
    --vocab_file=mobilebert_data_origin/vocab.txt  \
    --bert_config_file=mobilebert_data_origin/bert_config.json \
    --init_checkpoint=mobilebert_data_origin/mobilebert_variables.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=4.0   \
    --output_dir=./output/result_dir


perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt

