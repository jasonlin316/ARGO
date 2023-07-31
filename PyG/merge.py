import json
import collections
import argparse

traces = []
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=2, help='number of processes')
parser.add_argument('--l', type=int, default=1, help='number of loaders')
p = parser.parse_args().p
l = parser.parse_args().l

for i in range(p):
    with open(f'./DDP_profile/trace_p{p}_l{l}_r{i}.json') as f:
        traces.append(json.load(f))

# 合并所有trace事件到一个列表
all_events = []
for pid,trace in enumerate(traces):
    for event in trace:
        # print(event)
        event['pid'] = pid
        # print("new event: ", event)
    all_events.extend(trace)
  
# 排序  
all_events.sort(key=lambda e: e['ts'])

# 根据pid分组
grouped_events = collections.defaultdict(list)
for event in all_events:
  grouped_events[event['pid']].append(event)
  
# 构建新的trace  
new_trace = []
for pid, events in grouped_events.items():
  new_trace.extend(events)
  
# 写入文件
with open(f'DDP_profile/combine_p{p}_l{l}.json', 'w') as f:
  json.dump(new_trace, f)