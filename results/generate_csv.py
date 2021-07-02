#!/usr/bin/python3.8
#!/usr/bin/python3.8

import os, json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description = 'Description')
parser.add_argument('-a', help='factor for power law distribution',default='', type=str)
args = parser.parse_args()


addition = args.a
df  = pd.DataFrame()
experiments = os.listdir('./script'+addition)
for e in experiments:
    workflows = os.listdir("./script"+addition+f"/{e}/")
    for w in workflows:
        f = open("./script"+addition+f"/{e}/{w}")
        metrics = json.load(f)
        row = {
            "experiment": e,
            "workflow": metrics['metadata']['name'],
            "startedAt": metrics['status']['startedAt'],
            "finishedAt": metrics['status']['finishedAt'],
            "resourcesDuration_cpu": metrics['status']['resourcesDuration']['cpu'],
            "resourcesDuration_memory": metrics['status']['resourcesDuration']['memory'],
        }
        df = df.append(row, ignore_index=True)

df['duration'] = pd.to_datetime(df['finishedAt'], infer_datetime_format=True) - pd.to_datetime(df['startedAt'], infer_datetime_format=True)
df['duration'] = df['duration'].dt.seconds
df = df[["experiment", "workflow", "startedAt", "finishedAt", "duration", "resourcesDuration_cpu", "resourcesDuration_memory"]]
df.to_csv("./results"+addition+".csv",index=False)


