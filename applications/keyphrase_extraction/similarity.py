import os
import argparse
import functools
import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from utils import preprocess_function, read_local_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", required=True, default=None, type=str, help="Local dataset directory should include data.txt and label.txt")
parser.add_argument("--output_file", default="output.txt", type=str, help="Save prediction result")
parser.add_argument('--model_name', default="ernie-3.0-medium-zh", help="Select model to train, defaults to ernie-3.0-medium-zh.",
                    choices=["ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"])
parser.add_argument("--params_path", default="./checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--data_file", type=str, default="data.txt", help="Unlabeled data file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
args = parser.parse_args()
# yapf: enable


def cosine_sim(query_embedding, target_embedding):

    cosine_sim = paddle.matmul(query_embedding, target_embedding)

    return cosine_sim


def split_tokens(sequence, n_gram=2):

    assert isinstance(sequence, paddle.Tensor) and len(sequence.shape) == 3

    ids_splits = []

    for i in range(1, n_gram + 1):
        for n in range(sequence.shape[1] - i):
             sub_sequence = sequence[:, n : n + i, :]
             ids_splits.append([i, sub_sequence])

    return ids_splits


@paddle.no_grad()
def predict():
    """
    Predicts the data labels.
    """
    paddle.set_device(args.device)
    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    label_list = []
    label_path = os.path.join(args.dataset_dir, args.label_file)
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            label_list.append(line.strip())

    data_ds = load_dataset(
        read_local_dataset, path=os.path.join(args.dataset_dir, args.data_file), is_test=True, lazy=False
    )

    trans_func = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        label_nums=len(label_list),
        is_test=True,
    )
    
    data_ds = data_ds.map(trans_func)
    # print(dir(data_ds), data_ds.new_data, data_ds.info)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(data_ds, batch_size=args.batch_size, shuffle=False)

    data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)

    results = []
    model.eval()
    for batch in data_data_loader:
        logits = model(**batch)
        
        for logit in logits:
            text_embeddings = paddle.sum(logit, axis=-1)
        
        text_embedding = paddle.mean(logit, axis=0)

        # token_embedding = 
        # cos_similarity = F.cosine_similarity
        # for 
            # print(text_embeddings.shape)
            
        return
        # probs = F.sigmoid(logits).numpy()

    #     print()
    #     for prob in probs:
    #         labels = []
    #         for i, p in enumerate(prob):
    #             if p > 0.5:
    #                 labels.append(i)
    #         results.append(labels)

    # with open(args.output_file, "w", encoding="utf-8") as f:
    #     f.write("text" + "\t" + "label" + "\n")
    #     for d, result in zip(data_ds.data, results):
    #         label = [label_list[r] for r in result]
    #         f.write(d["sentence"] + "\t" + ", ".join(label) + "\n")
    # logger.info("Prediction results save in {}.".format(args.output_file))

    return


if __name__ == "__main__":

    # predict()

    x = paddle.randn(shape = [2,3,3])
    splits = split_tokens(x)
    print(splits)