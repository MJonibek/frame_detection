import os
from tqdm import tqdm
import pandas as pd
import numpy as np


def make_dataframe(self, input_folder, labels_folder=None):
        text = []

        for fil in tqdm(filter(lambda x: x.endswith('.txt'),
                           os.listdir(input_folder))):
            iD, txt = fil[7:].split('.')[0], open(os.path.join(input_folder, fil),
                                              'r', encoding='utf-8').read()
            text.append((iD, txt))

        df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')
        df = df_text

        if labels_folder:
            labels = pd.read_csv(labels_folder, sep='\t', header=None)
            labels = labels.rename(columns={0:'id', 1:'frames'})
            labels.id = labels.id.apply(str)
            labels = labels.set_index('id')

            df = labels.join(df_text)[['text', 'frames']]

        return df

