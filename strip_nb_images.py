import json, os

with open('notebooks/eda.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

HEAVY = {'image/png', 'image/jpeg', 'image/svg+xml', 'application/vnd.plotly.v1+json'}

cells_cleaned = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_outputs = []
    for out in cell.get('outputs', []):
        data = out.get('data', {})
        heavy_keys = set(data.keys()) & HEAVY
        light_keys = set(data.keys()) - HEAVY
        # Drop entire output only if it is nothing but heavy mime types
        if heavy_keys and not light_keys and out.get('output_type') not in ('stream', 'error'):
            continue
        for k in list(data.keys()):
            if k in HEAVY:
                del data[k]
        new_outputs.append(out)
    if len(new_outputs) != len(cell.get('outputs', [])):
        cells_cleaned += 1
    cell['outputs'] = new_outputs

with open('notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

size_mb = os.path.getsize('notebooks/eda.ipynb') / 1_000_000
print(f'Cleaned {cells_cleaned} cells with heavy outputs removed')
print(f'Final notebook size: {size_mb:.2f} MB')
