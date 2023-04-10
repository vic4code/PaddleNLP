import os
import argparse
import functools
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from utils import preprocess_function, read_local_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", default="data", type=str, help="Local dataset directory should include data.txt and label.txt")
parser.add_argument("--output_file", default="output.txt", type=str, help="Save prediction result")
parser.add_argument('--model_name', default="ernie-3.0-medium-zh", help="Select model to train, defaults to ernie-3.0-medium-zh.",
                    choices=["ernie-1.0-large-zh-cw", "ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en", "ernie-m-base", "ernie-m-large"])
parser.add_argument("--params_path", default="./checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--data_file", type=str, default="data.txt", help="Unlabeled data file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
args = parser.parse_args()
# yapf: enable

def standarize(x):
    return (x - paddle.mean(x))/(paddle.std(x) + 0.0001)

class cosine_sim(nn.Layer):
    def __init__(self, dropout=None):

        super().__init__()

    def get_pooled_embedding(self, embedding):

        embedding = F.normalize(embedding, p=2, axis=-1)

        return embedding

    def forward(self, query_embedding, target_embedding):

        # Get the mean of a subsequence

        #query_embedding += 10
        #target_embedding += 10

        #query_embedding = paddle.log10(query_embedding)
        #target_embedding = paddle.log10(target_embedding)

        query_embedding = paddle.mean(query_embedding, axis=1)

        query_embedding = self.get_pooled_embedding(query_embedding)

        target_embedding = self.get_pooled_embedding(target_embedding)
        
        cosine_sim = np.corrcoef(query_embedding.squeeze(0), target_embedding.squeeze(0))
        #cosine_sim = paddle.sum(query_embedding * target_embedding, axis=-1)
        return cosine_sim[0, 1]


def split_embeddings(embedding, batch, n_gram=4):

    assert n_gram <= embedding.shape[1]
    assert isinstance(embedding, paddle.Tensor) and len(embedding.shape) == 3

    embedding_splits = []

    for i in range(1, n_gram + 1):
        for n in range(1, embedding.shape[1] - i + 1):
            sub_embedding = embedding[:, n : n + i, :]
            embedding_splits.append(
                {
                    'n_gram': i,
                    'sub_embedding': sub_embedding,
                    'start': n,
                    'end': n + i,
                    'input_ids': batch['input_ids'][0, n:(n + i)],
                    'sim': 0
                }
            )
    
    return embedding_splits


@paddle.no_grad()
def predict():
    """
    Predicts the data labels.
    """
    paddle.set_device(args.device)
    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    label_list = []
    label_path = os.path.join("./applications/keyphrase_extraction/", args.dataset_dir, args.label_file)
    
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            label_list.append(line.strip())

    data_ds = load_dataset(
        read_local_dataset, path=os.path.join("./applications/keyphrase_extraction/", args.dataset_dir, args.data_file), is_test=True, lazy=False
    )

    #luka
    args.max_seq_length = len(data_ds[0]['sentence'])
    args.batch_size = 1

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

    similarity = cosine_sim()

    for batch in data_data_loader:

        sequence_outputs, pooled_outputs = model(**batch)
        target_embedding = sequence_outputs[:, 0, :][0].unsqueeze(0)

        sequence_outputs = sequence_outputs[0].unsqueeze(0)
        embeddings = split_embeddings(sequence_outputs, batch, n_gram=20)
        #tmp_sim = []
        for split in range(len(embeddings)):
            sim = similarity(embeddings[split]['sub_embedding'], target_embedding)
            #tmp_sim.append(sim)
            embeddings[split]['sim'] = sim
            #if sim[0] > 0.9953:
            #    output = tokenizer.convert_ids_to_tokens(batch['input_ids'][0, embeddings[split]['start']:embeddings[split]['end']])
                #print(output)
                #if '。' not in output:
                #    print(output)
        print(123)
        sorted_embeddings = sorted(embeddings, key=lambda x:x['sim'], reverse=True)

        for i in range(20):
            print(tokenizer.convert_ids_to_tokens(sorted_embeddings[i]['input_ids']))
        print("end")
        
        print(sorted_embeddings)

            # results.append(sim)

    # print(results)
    return


if __name__ == "__main__":

    predict()

    # x = paddle.randn(shape = [2, 10, 3])
    # splits = split_embeddings(x)
    # print(splits)
