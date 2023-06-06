import json
import re
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_file", default="doccano.jsonl", type=str, help="The doccano file exported from doccano platform.")
parser.add_argument("--save_dir", default="./data/jsonl/formal_data.jsonl", type=str, help="The path of data that you wanna save.")
args = parser.parse_args()

def jsonl_to_model_inputs():
    data_path = "data/formal_dataset/classification"
    data_files = [filename for filename in Path(data_path).rglob('*.json')]

    # data_files = [
    #                 # 'data/formal_dataset/文本標記/標記完成的匯出檔案-謹丞(NER)/文本分類0505.json',
    #                 # 'data/formal_dataset/文本標記/標記完成的匯出檔案-謹丞(NER)/文本分類0508.json',
    #                 'data/formal_dataset/文本標記/標記完成的匯出檔案-謹丞(NER)/文本分類0509.json'
    #             ]

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

if __name__ == "__main__":
    jsonl_to_model_inputs()