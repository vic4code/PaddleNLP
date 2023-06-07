import json
import re
import argparse
import os
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_file", default="doccano.jsonl", type=str, help="The doccano file exported from doccano platform.")
parser.add_argument("--save_dir", default="./data/jsonl/formal_data.jsonl", type=str, help="The path of data that you wanna save.")
args = parser.parse_args()

def json_to_jsonl():
    data_path = "data/formal_dataset/classification"
    data_files = [filename for filename in Path(data_path).rglob('*.json')]

    print("Num files : ", len(data_files))
    whitespace = r"\s+"

    id = set()
    err_count = 0
    preprocessed_data = []

    for data_file in data_files:

        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        for data in raw_data:
            try:
                if data['annotations'][0]['result'][0]['id'] not in id:
                    id.add(data['annotations'][0]['result'][0]['id'])
                    preprocessed_data.append( 
                        {
                            'id': data['annotations'][0]['result'][0]['id'],
                            'jid': data['data']['jid'],
                            'data': re.sub(whitespace, "", data['data']['text']),
                            'label': data['annotations'][0]['result'][0]['value']['choices']
                        }
                    )
            except:
                err_count += 1

    print("Total number of data", len(preprocessed_data))
    print("Error files: ", err_count)

    with open(args.save_dir, 'w', encoding="utf-8") as f:
        for item in preprocessed_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def gen_toydata():
    data_file = "data/jsonl/formal_data.jsonl"
        
    table = {"18-24歲(大學、研究所)",
            "25-29歲",
            "30-39",
            "40-49歲",
            "50-59歲",
            "60-64歲",
            "65歲以上(退休)",
            "上肢",
            "下肢",
            "休克",
            "內出血",
            "其他",
            "壓迫",
            "壞死",
            "失能",
            "截肢",
            "扭傷",
            "損傷",
            "撕裂傷",
            "擦挫傷",
            "未滿18歲(高中以下)",
            "栓塞",
            "死亡",
            "灼傷",
            "瘀血",
            "破缺損",
            "神經損傷",
            "肇責 0/100",
            "肇責 10/90",
            "肇責 100/0",
            "肇責 20/80",
            "肇責 30/70",
            "肇責 40/60",
            "肇責 50/50",
            "肇責 60/40",
            "肇責 70/30",
            "肇責 80/20",
            "肇責 90/10",
            "背部",
            "胸部",
            "胸部損傷",
            "脫位",
            "腦震盪",
            "腹部",
            "臉",
            "衰竭",
            "鈍傷",
            "頭頸部",
            "骨折",
            "骨盆",
            "骨裂",
            "拉傷"}
    sub_table = {
            "上肢",
            "下肢",
            "背部",
            "胸部",
            "胸部損傷",
            "脫位",
            "腦震盪",
            "腹部",
            "臉",
            "衰竭",
            "鈍傷",
            "頭頸部",
            "骨折",
            "骨盆",
            "骨裂",
            "拉傷"}

    filter_data = []

    with open(data_file,  "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        json_obj = json.loads(line)
        for i, label in enumerate(json_obj["label"]):
            if label not in sub_table:
                continue

            if i == len(json_obj["label"]) - 1:
                filter_data.append(json_obj)
    
    random.seed()
    sample_toy = filter_data
    # sample_toy = random.choices(filter_data, k=20)
    # breakpoint()
    with open("data/jsonl/toy_samples.jsonl", 'w', encoding="utf-8") as f:
        for item in sample_toy:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def rule_based_truncate(data_file="data/jsonl/formal_data"):
    
    whitespace = r"\s+"
    data_file = "data/jsonl/formal_data.jsonl"

    with open(data_file,  "r", encoding="utf-8") as f:
        lines = f.readlines()
        preprocessed_data = []
        label_studio_format = []

        for line in lines:
            json_obj = json.loads(line)

            id = json_obj['id']
            start_index = 0
            text = json_obj['data']

            print(id)

            if text.find("受有") == -1:
                print(f"Id {id} has no 受有, 傷害")
                continue
            
            # Collect start indices
            start_indices = []

            while start_index != -1:

                start_index = text.find("受有", start_index)

                if start_index != -1:
                    start_indices.append(start_index)
                    start_index += 2

            # Find end index
            output_text = ""
            for start_index in start_indices:
                end_index = text[start_index:].find("傷害") + start_index - 1

                if end_index > start_index and end_index - start_index < 70 and end_index - (start_index + 2) > 1:
                    output_text = text[start_index + 2:end_index]

                    break
            
            if not output_text:
                for start_index in start_indices:
                    end_index = text[start_index:].find("傷勢") + start_index - 1

                    if end_index > start_index and end_index - start_index < 70 and end_index - (start_index + 2) > 1:
                        output_text = text[start_index + 2:end_index]
                        break
                
            # Save as clean injury classification data
            label_list = json_obj['label']
            labels = [label for label in label_list if "歲" not in label and "肇責" not in label]
            if output_text:
                preprocessed_data.append( 
                    {
                        'id': json_obj['id'],
                        'jid': json_obj['jid'],
                        'data': output_text,
                        'label': labels
                    }
                )

            # breakpoint()

            # label-studio format
            label_studio_format.append( 
                        {
                            "annotations": [
                                {
                                    "result": [
                                        {
                                            "value":{
                                                "choices":labels
                                            }

                                        }
                                    ]
                                }

                            ],
                            "data": {
                                "text":output_text
                            } 
                        }
                    )


        # Save as jsonl
        with open("data/jsonl/injury.jsonl", 'w', encoding="utf-8") as f:
            for item in preprocessed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        # Save as json
        with open("data/jsonl/injury.json", "w", encoding="utf-8") as f:
            json.dump(label_studio_format, f, ensure_ascii=False)


if __name__ == "__main__":
    # print("Collecting json files to save as jsonl")
    # json_to_jsonl()
    # gen_toydata()

    rule_based_truncate()