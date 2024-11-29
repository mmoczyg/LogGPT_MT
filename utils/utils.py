# © 2023 Nokia
# Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import random
import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import regex as re
from tqdm import tqdm
from ast import literal_eval
import json


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def hdfs_blk_process(df, blk_label_dict):
    data_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in data_dict:
                data_dict[blk_Id] = [row['EventId']]
            else:
                data_dict[blk_Id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

    data_df["Label"] = data_df["BlockId"].apply(
        lambda x: blk_label_dict.get(x))  # add label to the sequence of each blockid

    return data_df


def sliding_window(df, options):
    if options is None:
        options = {
            "dataset_name": "MtController",
            "window_size": 100,  # Größe des Sliding Windows
            "step_size": 50,     # Schrittweite des Sliding Windows
            "max_lens": 10       # Maximale Länge eines Sliding Windows
        }
    else:
        # Standardwerte setzen, falls Schlüssel fehlen
        options.setdefault("dataset_name", "MtController")
        options.setdefault("window_size", 100)
        options.setdefault("step_size", 50)
        options.setdefault("max_lens", 10)

    print(f"Dataset: {options['dataset_name']}")
    print(f"DataFrame Größe: {df.shape}")
    print(f"Spalten: {df.columns.tolist()}")

    if options['dataset_name'] == 'MtController':
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datatime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
        else:
            raise KeyError("The required columns 'Date' and 'Time' are missing.")
    
    df['timestamp'] = df['datatime'].values.astype(np.int64) // 10 ** 9
    df = df.sort_values('timestamp')
    df.set_index('timestamp', drop=False, inplace=True)
    
    start_time = df.timestamp.min()
    end_time = df.timestamp.max()

    print(f"Startzeit: {start_time}, Endzeit: {end_time}")

    new_data = []
    while start_time < end_time:
        df_window = df.loc[start_time:start_time + options["window_size"]]
        if len(df_window) > 1:
            if len(df_window) > options['max_lens']:
                start_time_inner = df_window.timestamp.min()
                end_time_inner = df_window.timestamp.max()
                while (end_time_inner - start_time_inner) > options['max_lens']:
                    df_window_inner = df_window.loc[start_time_inner:start_time_inner + options['max_lens']]
                    new_data.append([
                        df_window_inner['Label'].values.tolist(),
                        df_window_inner['Label'].max(),
                        df_window_inner['msg'].values.tolist()
                    ])
                    start_time_inner += options['max_lens'] // 2
            else:
                new_data.append([
                    df_window['Label'].values.tolist(),
                    df_window['Label'].max(),
                    df_window['msg'].values.tolist()
                ])
        start_time += options['step_size']

    print(f"Erzeugte Sliding Windows: {len(new_data)}")
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])


def preprocessing(preprocessing=True, dataset_name='MtController', options=True):
    if preprocessing:
        if dataset_name == 'MtController':
            print("Preprocessing MtController dataset")
            input_file = './datasets/mt_controller_log.json_structured.csv'
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_file}")

            df = pd.read_csv(input_file)
            print(f"Datei geladen mit {len(df)} Zeilen und {len(df.columns)} Spalten.")
            df.rename(columns={'date': 'Date', 'time': 'Time'}, inplace=True)

            required_columns = ['date', 'time', 'msg', 'severity']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Erforderliche Spalte '{col}' fehlt im Datensatz.")

            print("Erstelle Zeitstempel...")
            df['datatime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
            df['timestamp'] = df['datatime'].values.astype(np.int64) // 10**9

            if 'Label' not in df.columns:
                print("Label-Spalte fehlt. Füge Dummy-Labels hinzu...")
                df['Label'] = 0  # Dummy-Wert für Labels

            print("Starte Sliding-Window-Verarbeitung...")
            new_df = sliding_window(df, options)

            if new_df.empty:
                print("Fehler: Keine Sliding Windows erzeugt. Prüfen Sie die Eingabedaten und Optionen.")
                return

            output_file = f'./datasets/MtController.W{options["window_size"]}.S{options["step_size"]}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            new_df.to_csv(output_file, index=False)

            if os.path.exists(output_file):
                print(f"Datei erfolgreich erstellt: {output_file}")
            else:
                print("Fehler: Datei wurde nicht erstellt.")


def train_test_split(dataset_name='MtController', train_samples=5000, seed=42, options=None, dir='.'):
    if dataset_name == 'MtController':
        df = pd.read_csv(f'{dir}/datasets/MtController.W{options["window_size"]}.S{options["step_size"]}.csv')
        df['EventSequence'] = df['EventSequence'].apply(literal_eval)
        train_df = df.sample(frac=0.8, random_state=seed).reset_index(drop=True)
        test_df = df.drop(train_df.index).reset_index(drop=True)
        train_df.to_csv(f'{dir}/datasets/MtController.train.csv', index=False)
        test_df.to_csv(f'{dir}/datasets/MtController.test.csv', index=False)
        print(f"Training dataset: {len(train_df)} entries")
        print(f"Testing dataset: {len(test_df)} entries")
        return train_df, test_df


def get_training_dictionary(df):
    dic = {}
    count = 0
    for i in range(len(df)):
        lst = list(df['EventSequence'].iloc[i])
        for j in lst:
            if j in dic:
                pass
            else:
                dic[j] = str(count)
                count += 1
    return dic
