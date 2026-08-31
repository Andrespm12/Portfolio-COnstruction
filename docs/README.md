# docs/

Sources for the two human-facing artifacts. Both were living in a throwaway
scratchpad, which meant the documentation about a risk model had no version
history and no way to tell whether it still matched the code. That is the
failure mode the master document itself warns about in its closing line, so
they belong here.

| File | What it is |
|:--|:--|
| `documento_maestro.html` | The master document — how the model works, end to end, in Spanish. Import into Google Docs (Drive converts HTML on upload) or open in a browser. |
| `presentacion_cci.js` | Builds the 34-slide CCI-branded deck with pptxgenjs. |
| `auditar_pptx.py` | Geometric audit of a built PPTX: shapes off-slide, text colliding with the footer, estimated text overflow. |

## Building the deck

```
cd docs && npm install && node presentacion_cci.js
python auditar_pptx.py CCI_modelo.pptx
```

pptxgenjs writes the file nearly uncompressed (~960 KB). Repack it before
sending — same bytes, a sixth of the size:

```python
import zipfile
zin = zipfile.ZipFile("CCI_modelo.pptx")
with zipfile.ZipFile("deck.pptx", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        item.compress_type = zipfile.ZIP_DEFLATED   # source entries are STORED
        zout.writestr(item, data)
```

`auditar_pptx.py` exists because LibreOffice cannot open a PPTX in this
environment at all — it fails on a trivial two-shape file — so there is no way
to render one and look at it. The audit reads the drawing XML and checks
geometry instead. Its text-overflow check is an estimate from character count,
font size and box dimensions, so treat a "texto justo" warning as a prompt to
look, not as proof; the margin warnings are cosmetic.

## Keeping these honest

Both files carry numbers measured from the code: the 447-name candidate
universe, the equilibrium-anchor table, the test count. When any of those
change, these change too. The anchor table is reproducible:

```
python -c "
import sys; sys.path[:0] = ['.', 'tests']
from test_optimizer import world
..."
```

— re-solve `world()` from `tests/test_optimizer.py` under both anchors and read
off total equity weight. It is fixture output, not a forecast, and both
documents say so where they print it.
