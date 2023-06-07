# Verdict NLP applications

## Installation

### cpu version
```
pip install paddlepaddle
pip install paddlepaddle-gpu
python3 -m  pip install scikit-learn==1.0.2
```

### gpu version
```
conda create -n paddle python=3.7
pip install --upgrade paddlenlp
# pip install paddlepaddle-gpu
# python3 -m  pip install scikit-learn==1.0.2
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
pip install git+https://github.com/PaddlePaddle/PaddleNLP.git
```

### Install by git source code
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
export PYTHONPATH=/path/to/your/module/directory
export PYTHONPATH=/home/ubuntu/projects/PaddleNLP
pip install paddlepaddle
pip install -r requirements.txt     
```

## Text classfication
### Data preprocess
```
python data_preprocess.py \
--raw_data_dir "data/formal_dataset/xxx.json"git@github.com:vic4code/verdict.git
```

### Convert jsonl to data splits for training
```
python data_split.py \
    --jsonl_file ./data/jsonl/data_108.jsonl \
    --save_dir ./data/dataset \
    --splits 0.8 0.1 0.1 \
    --task_type "multi_label"
```


### Finetune
```
cd textclassfication/finetune
python train.py \                                                     
--dataset_dir "data/dataset" \
--device "cpu" \
--max_seq_length 128 \
--model_name "ernie-3.0-medium-zh" \
--batch_size 32 \
--early_stop \
--epochs 10
```
### Few-shot prompt learning
```
cd textclassfication/few-shot
python train.py \
--data_dir ./data/dataset \
--output_dir ./checkpoints/ \
--prompt "這句話要包含的要素有" \
--model_name_or_path ernie-3.0-base-zh \
--max_seq_length 2048  \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--do_train \
--do_eval \
--do_predict \
--do_export \
--num_train_epochs 100 \
--logging_steps 5 \
--save_total_limit 1 \
--per_device_eval_batch_size 1 \
--per_device_train_batch_size 1 \
--metric_for_best_model micro_f1_score \
--load_best_model_at_end \
--evaluation_strategy epoch \
--save_strategy epoch
```

### cpu version
--dataloader_drop_last = True to frop the uncomplete batch, but it's not valid when testing.
```
python train.py \
--device "cpu" \
--data_dir ./data/toy_dataset \
--output_dir ./checkpoints/ \
--prompt "這句話要包含的要素有" \
--model_name_or_path ernie-3.0-base-zh \
--max_seq_length 256  \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--do_train \
--do_eval \
--do_predict \
--do_export \
--num_train_epochs 100 \
--logging_steps 5 \
--save_total_limit 1 \
--per_device_eval_batch_size 8 \
--per_device_train_batch_size 8 \
--metric_for_best_model micro_f1_score \
--load_best_model_at_end \
--evaluation_strategy epoch \
--save_strategy epoch \
--dataloader_drop_last True
```

### Custom model (XLNet)
```
python train.py \
--device "cpu" \
--data_dir ./data/dataset \
--output_dir ./checkpoints/ \
--prompt "這句話要包含的要素有" \
--model_name_or_path chinese-xlnet-base \
--max_seq_length 2048 \
--mem_len 512 \
--reuse_len 512 \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--do_train \
--do_eval \
--do_predict \
--do_export \
--num_train_epochs 100 \
--logging_steps 5 \
--save_total_limit 1 \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size 8 \
--metric_for_best_model micro_f1_score \
--load_best_model_at_end \
--evaluation_strategy epoch \
--save_strategy epoch
```

### XLNet gpu
``` 
python -u -m paddle.distributed.launch --gpus 0,1 train.py --data_dir ./data/dataset --output_dir ./checkpoints/ --prompt "這句話要包含的要素有" --model_name_or_path chinese-xlnet-base --max_seq_length 2048 --learning_rate 3e-5 --ppt_learning_rate 3e-4 --do_train --do_eval --do_predict --do_export --num_train_epochs 100 --logging_steps 5 --save_total_limit 1 --per_device_eval_batch_size 1 --per_device_train_batch_size 1 --metric_for_best_model micro_f1_score --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch
```

### Ernie on instance
```
python train.py --data_dir ./data/dataset --output_dir ./checkpoints/ --prompt "這句話要包含的要素有" --model_name_or_path ernie-3.0-base-zh --max_seq_length 64 --learning_rate 3e-5 --ppt_learning_rate 3e-4 --do_train --do_eval --do_predict --do_export --num_train_epochs 10 --logging_steps 5 --save_total_limit 1 --per_device_eval_batch_size 1 --per_device_train_batch_size 2 --metric_for_best_model micro_f1_score --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch  --dataloader_drop_last True
```
