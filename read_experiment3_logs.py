import os
import glob
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from os.path import dirname, basename
from google.protobuf.json_format import MessageToDict

df = pd.DataFrame({'wallTime': pd.Series(dtype='int64')})

for logfile in glob.glob("logs/experiment3/*/*/*/events.out*"):
    d = dirname(dirname(logfile))
    epoch = basename(d).split('-')[1]
    exp = basename(dirname(d))
    for r in summary_iterator(logfile):
        m = MessageToDict(r)
        if 'summary' in m:
            walltime = m['wallTime']
            summary = m['summary']
            if 'value' in summary and len(summary['value'])>=1:
                value = summary['value'][0]
                tag = value['tag']
                if 'simpleValue' in value:
                    value = value['simpleValue']
                    df = df.append({
                        'exp': exp,
                        'epoch': epoch,
                        'wallTime': int(walltime),
                        'tag': tag,
                        'value': value
                        }, ignore_index=True)
df.to_csv('experiment3_logs.csv')
