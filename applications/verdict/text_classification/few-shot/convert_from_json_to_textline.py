import pandas as pd
import json
import re
import random
from pathlib import Path

random.seed(10)

data_path = '202301'
whitespace = r"\s+"

filenames = [filename for filename in Path(data_path).rglob('*.json')]
samples = random.sample(filenames, 15)

for n, sample in enumerate(samples):
    f = open(sample)
    data = json.load(f)
    text = re.sub(whitespace, "", data['JFULL'])
    file = open(Path('data')/(str(n) + '.txt'), "w")
    a = file.write(text)
    file.close()

