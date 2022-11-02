import os
import csv

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

cpath = os.getcwd()

def tabulate_events(dpath):

    for dname in os.listdir(dpath):
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['scalars']
        out = {}
        
        for tag in tags:
            tag_values=[]
            for event in ea.Scalars(tag):
                tag_values.append(event.value)
            out[tag] = tag_values[:-7]

        out_keys = [k for k in out.keys()]
        out_values = [v for v in out.values()]
        
        for i in range(2):
            try:
                with open(f"../result/{out_keys[i]}.csv", 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(out_values[i])
            except IndexError:
                print(f'{tag}')

    return "Converted"

cpath = os.getcwd()
folderpath = os.chdir(cpath + "/runs")

for folder in os.listdir(folderpath):
    path = f'{folder}'
    tabulate_events(path)

