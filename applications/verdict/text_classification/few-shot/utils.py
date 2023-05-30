# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from paddlenlp.datasets import load_dataset


def load_local_dataset(data_path, splits, label_list, train_batch_size, eval_batch_size, chunk_len=None, prompt=None, other_tokens_length=None, overlap_length=None):
    """
    Load dataset for multi-label classification from files, where
    there is one example per line. Text and label are seperated
    by '\t', and multiple labels are delimited by ','.

    Args:
        data_path (str):
            Path to the dataset directory, including label.txt, train.txt,
            dev.txt (and data.txt).
        splits (list):
            Which file(s) to load, such as ['train', 'dev', 'test'].
        label_list (dict):
            The dictionary that maps labels to indeces.
    """

    def chunker(text, chunk_len, batch_size, other_tokens_length, overlap_length):
        # other_tokens: [CLS], [MASK], [SEP]
        if chunk_len:
            sequence_length = len(text)
            # divider = chunk_len - len(prompt) - other_tokens_length
            divider = chunk_len
            num_chunks = sequence_length // divider
            chunks = []

            i = 0
            while i < num_chunks:
                start, end = i * divider - overlap_length if i * divider - overlap_length >= 0 else i * divider , (i + 1) * divider
                chunks.append(text[start:end])
                i += 1
            
            remainder = len(chunks) % batch_size
            for i in range(remainder):
                chunks.append("")

            return chunks


    def _reader(data_file, label_list, batch_size, chunk_len=chunk_len, overlap_length=overlap_length):
        with open(data_file, "r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                data = line.strip().split("\t")
                if len(data) == 1:
                    yield {"text_a": data[0]}
                elif chunk_len:
                    text, label = data
                    label = label.strip().split(",")
                    label = [float(1) if x in label else float(0) for x in label_list]
                    chunks = chunker(text, chunk_len, batch_size, other_tokens_length, overlap_length)

                    for nth, chunk in enumerate(chunks) :
                        yield {"id": idx , "nth_chunk": nth, "num_chunks": len(chunks) - 1, "text_a": chunk, "labels": label}
                else:
                    text, label = data
                    label = label.strip().split(",")
                    label = [float(1) if x in label else float(0) for x in label_list]
                    yield {"text_a": text, "labels": label}

    split_map = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
    datasets = []
    for split in splits:
        data_file = os.path.join(data_path, split_map[split])
        if split == "train":
            datasets.append(load_dataset(_reader, data_file=data_file, label_list=label_list, batch_size=train_batch_size, lazy=False))
        else:
            datasets.append(load_dataset(_reader, data_file=data_file, label_list=label_list, batch_size=eval_batch_size, lazy=False))
    return datasets
