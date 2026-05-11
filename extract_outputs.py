"""
extract_outputs.py

When running lux_dna_to_SD.ipynb on Google Colab, generated images are embedded
directly in the notebook's cell outputs as base64-encoded data rather than saved
to disk. This script reads the .ipynb file locally and extracts every embedded
image to extracted_outputs/ so the results are available without having to
manually download each one from Colab.

Usage (run locally after downloading the notebook from Colab):
    python3 extract_outputs.py
"""

import json
import base64
import os

NOTEBOOK = 'lux_dna_to_SD.ipynb'
OUT_DIR  = 'extracted_outputs'

MIME_EXT = {
    'image/png':  '.png',
    'image/jpeg': '.jpg',
    'image/gif':  '.gif',
    'image/svg+xml': '.svg',
}

os.makedirs(OUT_DIR, exist_ok=True)

with open(NOTEBOOK) as f:
    nb = json.load(f)

saved = 0

for cell_idx, cell in enumerate(nb['cells']):
    for out_idx, output in enumerate(cell.get('outputs', [])):
        data = output.get('data', {})
        for mime, ext in MIME_EXT.items():
            if mime not in data:
                continue
            raw = data[mime]
            # outputs store base64 as either a single string or a list of strings
            b64 = raw if isinstance(raw, str) else ''.join(raw)
            filename = f'cell_{cell_idx:03d}_output_{out_idx}{ext}'
            path = os.path.join(OUT_DIR, filename)
            with open(path, 'wb') as img_file:
                img_file.write(base64.b64decode(b64))
            print(f'  saved {path}')
            saved += 1
            break  # one image type per output is enough

print(f'\nDone — {saved} images saved to {OUT_DIR}/')
