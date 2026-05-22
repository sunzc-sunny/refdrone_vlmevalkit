import sys, os
sys.path.insert(0, '.')
os.environ['LMUData'] = '/mnt/tidal-alsh01/dataset/perceptionVLM/benchmark/'

from vlmeval.smp import load, parse_file

# 1. 检查 TSV 结构
data = load('/mnt/tidal-alsh01/dataset/perceptionVLM/benchmark/refdrone_test.tsv')
print('=== TSV columns:', list(data.columns))
print('=== shape:', data.shape)

row = data.iloc[0]
print('=== first row keys with values:')
for k in data.columns:
    v = str(row[k])
    print(f'  {k}: {v[:120]}')

# 2. 模拟 dump_image 过程，看实际 image path 是什么
from vlmeval.dataset.image_grounding import Grounding_RefDrone

ds = Grounding_RefDrone(dataset='refdrone_test')
ds_data = ds.data
print('\n=== Dataset data columns:', list(ds_data.columns))

img_path = ds.dump_image(ds_data.iloc[0])
print('\n=== dump_image result:', img_path)

# 3. 用 parse_file 验证
if isinstance(img_path, list):
    for p in img_path:
        mime, s = parse_file(p)
        print(f'parse_file({p!r}) -> mime={mime}, exists={os.path.exists(p)}')
else:
    mime, s = parse_file(img_path)
    print(f'parse_file({img_path!r}) -> mime={mime}, exists={os.path.exists(img_path)}')
