import os
import glob
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from os.path import dirname, basename
from google.protobuf.json_format import MessageToDict

df = pd.DataFrame({'wallTime': pd.Series(dtype='int64')})

epoch_add = {
}

for logfile in glob.glob("logs/experiment1/*/*/*/events.out*"):
    d = dirname(dirname(dirname(logfile)))
    dtstamp = basename(d)
    for r in summary_iterator(logfile):
        m = MessageToDict(r)
        if 'summary' in m and 'step' in m:
            walltime = m['wallTime']
            step = m['step']
            summary = m['summary']
            if 'value' in summary and len(summary['value'])>=1:
                value = summary['value'][0]
                tag = value['tag']
                if 'simpleValue' in value:
                    value = value['simpleValue']
                    # if tag == 'epoch': value = value + epoch_add[dtstamp]
                    df = df.append({
                        'dtstamp': dtstamp,
                        'step': step,
                        'wallTime': int(walltime),
                        'tag': tag,
                        'value': value
                        }, ignore_index=True)
    break

loss = df[df.tag=='training_loss_step'].rename(columns={'value':'training_loss_step'}).drop(columns='tag')
epoch = df[df.tag=='epoch'].rename(columns={'value':'epoch'}).drop(columns='tag')
df = pd.merge(loss, epoch, on=['dtstamp','step','wallTime'])
df.to_csv('experiment1_logs.csv')
