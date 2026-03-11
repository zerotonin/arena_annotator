```
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀█░█▀▄░█▀▀░█▀█░█▀█░░░█▀█░█▀█░█▀█░█▀█░▀█▀░█▀█░▀█▀░█▀█░█▀▄░░         ║
 ║  ░█▀█░█▀▄░█▀▀░█░█░█▀█░░░█▀█░█░█░█░█░█░█░░█░░█▀█░░█░░█░█░█▀▄░░         ║
 ║  ░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀░▀░░         ║
 ║                                                                       ║
 ║   Interactive polygon & circle annotation for scientific arenas       ║
 ║   ── click. drag. export. science. ──                        v1.0     ║
 ╚═══════════════════════════════════════════════════════════════════════╝
```

# Arena Annotator

A single-file, dependency-light tool for annotating polygon and circular regions in experimental images. Built for the common workflow in behavioural neuroscience where you need to map arena boundaries from camera frames to real-world coordinates.

Click vertices onto your arena, drag to adjust, export to standard formats. Sessions are crash-proof and resumable. That's it.

| Tool | Shape | Use case |
|------|-------|----------|
| `arena_annotator.py` | Polygon | Rectangular, hexagonal, or irregular arenas |
| `circle_annotator.py` | Circle | Petri dishes, round open fields, circular mazes |

## Motivation

If you track animals in arenas — open fields, mazes, thermal gradient setups — you eventually need to define the arena boundary in pixel coordinates so you can project tracked positions into millimetres. This usually means writing a quick script with `cv2.setMouseCallback` for the twentieth time, losing the annotations when the script crashes, and then doing it again for 200 images.

Arena Annotator replaces that loop. It persists every click to disk immediately, resumes where you left off, lets you propagate annotations across frames with one keypress, and exports to the formats your downstream pipeline already expects.

## When to use this — and when not to

This tool does one thing: mark a fixed shape (polygon or circle) on a batch of images. If that's what you need, it will take you 30 seconds to install and less to learn.

If you need more — multi-class labelling, freehand masks, bounding boxes, team workflows, model-in-the-loop pre-annotation — use one of the full-featured annotation platforms:

| Tool | Scope | Notes |
|------|-------|-------|
| [CVAT](https://github.com/cvat-ai/cvat) | Image & video annotation | Open source (Intel). Interpolation, tracking, SAM integration. The current standard for serious CV annotation. Docker-based. |
| [Label Studio](https://github.com/HumanSignal/label-studio) | Multi-modal (image, text, audio, video) | Open source (HumanSignal). Very flexible config system. Good if you label more than just images. |
| [LabelImg](https://github.com/HumanSignal/labelImg) | Lightweight image annotation | The classic. Archived since Feb 2024, now folded into Label Studio. Still works for simple bounding-box tasks. |
| [LabelMe](https://github.com/labelmeai/labelme) | Polygonal image annotation | Open source. Closest in spirit to Arena Annotator but more general-purpose. Good if you need freehand polygons and multiple object classes. |
| [Roboflow](https://roboflow.com/) | End-to-end CV pipeline | Commercial, free tier. Annotation + augmentation + training + deployment in one platform. |

Arena Annotator fills the gap below all of these: when you just need a shape on each frame, you don't want Docker, and you don't want to register for anything.

## Installation

Three dependencies, all in every major conda channel and on PyPI:

```bash
pip install matplotlib numpy Pillow
```

Or with conda:

```bash
conda install matplotlib numpy pillow
```

Then just put `arena_annotator.py` and/or `circle_annotator.py` somewhere on your `$PATH`, or call them directly:

```bash
python arena_annotator.py --help
python circle_annotator.py --help
```

No Qt bindings, no OpenCV, no compiled extensions. Works on macOS, Linux, and Windows.

---

## Polygon Annotator — Quick start

Annotate a rectangular arena across a directory of frames:

```bash
python arena_annotator.py \
    -d ./trial_images/ \
    -v 4 \
    -l "TL,TR,BR,BL" \
    -a coco,yolo
```

This opens a matplotlib window. Left-click to place the four vertices in order, drag to adjust. Press → to advance to the next image. Press Q when done — COCO JSON and YOLO label files appear in `./trial_images/`.

### Polygon usage

```
arena_annotator [-h] (-d DIR | -i IMAGE | -f FILELIST)
                -v VERTICES [-l LABELS] [-o OUTPUT] [-a FORMATS]
```

| Flag | Description |
|------|-------------|
| `-d`, `--directory` | Directory of images (scans for png, jpg, tif, bmp, webp) |
| `-i`, `--image` | Single image file |
| `-f`, `--filelist` | Text file with one image path per line (`#` comments allowed) |
| `-v`, `--vertices` | Number of polygon vertices (≥ 3) |
| `-l`, `--labels` | Comma-separated vertex labels. Count must match `-v`. Default: `P1, P2, …` |
| `-o`, `--output` | Output directory. Default: same as image source |
| `-a`, `--formats` | Export formats, comma-separated: `coco`, `yolo`, `voc`. Default: `coco` |

### Polygon keybindings

| Input | Action |
|-------|--------|
| **Left click** | Place vertex (up to `-v` count) |
| **Left drag** on vertex | Move vertex |
| **Right click** on vertex | Delete vertex |
| ← / → | Previous / next image |
| **R** | Repeat vertices from the nearest preceding annotated image |
| **X** | Reset all vertices on current image (with Y/N confirmation) |
| **L** | Toggle vertex labels |
| **F** | Toggle polygon fill |
| **H** | Help overlay |
| **S** | Save & export now |
| **Q** / Esc | Save & quit |

---

## Circle Annotator — Quick start

Annotate a round arena across a directory of frames:

```bash
python circle_annotator.py \
    -d ./petri_dish_images/ \
    -a coco,yolo
```

This opens a matplotlib window. **First click** places the circle centre. **Second click** sets the radius (distance from centre to click). After that, **drag the centre** to reposition and **drag the rim handle** (cyan diamond at 3-o'clock) to resize. Press → to advance. Press Q when done.

### Circle usage

```
circle_annotator [-h] (-d DIR | -i IMAGE | -f FILELIST)
                 [-o OUTPUT] [-a FORMATS]
```

| Flag | Description |
|------|-------------|
| `-d`, `--directory` | Directory of images |
| `-i`, `--image` | Single image file |
| `-f`, `--filelist` | Text file with one image path per line |
| `-o`, `--output` | Output directory. Default: same as image source |
| `-a`, `--formats` | Export formats: `coco`, `yolo`, `voc`. Default: `coco` |

### Circle keybindings

| Input | Action |
|-------|--------|
| **1st left click** | Set circle centre |
| **2nd left click** | Set radius |
| **Drag centre** | Reposition circle |
| **Drag rim handle** | Resize circle |
| **Right click** | Delete annotation (clear centre + radius) |
| ← / → | Previous / next image |
| **R** | Repeat circle from nearest preceding annotated image |
| **X** | Reset circle (with Y/N confirmation) |
| **L** | Toggle labels (centre coords, radius) |
| **F** | Toggle circle fill |
| **H** | Help overlay |
| **S** | Save & export now |
| **Q** / Esc | Save & quit |

### Circle sidecar format

```json
{
  "image_path": "/data/experiment_01/frame_0042.png",
  "image_width": 640,
  "image_height": 480,
  "centre": [312.5, 245.8],
  "radius": 198.3
}
```

### Circle in export formats

The circle is approximated as a 64-sided polygon for COCO/YOLO/VOC export, so it works seamlessly with downstream pipelines that expect polygonal masks. The native circle parameters (centre, radius) are preserved in COCO `attributes` and VOC `<circle>` elements.

---

## Export formats

### COCO JSON

A single `annotations_coco.json` covering all images. For polygons: standard polygon segmentation with `attributes.vertex_labels`. For circles: polygon approximation with `attributes.shape`, `attributes.centre_x`, `attributes.centre_y`, `attributes.radius`.

### YOLO v8 polygon

One `.txt` per image in `yolo_labels/`. Coordinates normalised to `[0, 1]`. Class is always `0`.

### Pascal VOC XML

One `.xml` per image in `voc_annotations/`. Standard `<bndbox>` plus `<polygon>` (or `<circle>` + `<polygon>` for circle annotations).

## Session persistence

Every change is instantly written to a JSON sidecar file in the output directory. If the process crashes, the window is closed accidentally, or you come back the next day, the annotator picks up exactly where you left off.

## Design notes

**Single file.** No package structure, no `setup.py`, no build step. Copy the script, install three packages, go.

**matplotlib backend.** Chosen deliberately over OpenCV `highgui` or Qt for maximum portability. The only trade-off is that rendering during drag is not buttery smooth on very large images — but it's fine for the 640×480 to 2048×2048 range typical in behavioural setups.

**Repeat key (R).** In longitudinal experiments with a fixed camera, the arena barely moves between sessions. Press R to copy the annotation from the last annotated frame and adjust from there — typically saves 90% of the clicking.

## Documentation

API documentation is auto-generated from docstrings and published via GitHub Pages:

**[📖 Read the docs →](https://zerotonin.github.io/arena_annotator/)**

The docs are rebuilt automatically on every push to `main` via GitHub Actions.

## Author

**Bart R.H. Geurten**
Department of Zoology, University of Otago, Dunedin, New Zealand

- [University profile](https://www.otago.ac.nz/zoology/staff/dr-bart-geurten)
- [Google Scholar](https://scholar.google.de/citations?user=OAm7kgcAAAAJ&hl=en)
- [ORCID](https://orcid.org/0000-0002-1816-3241)

## License

[MIT](LICENSE)

## Citation

If this tool is useful in your published work, a citation or acknowledgement is appreciated. You can use the **"Cite this repository"** button on GitHub, or cite as:

```
Geurten, B. R. H. (2026). Arena Annotator: Interactive polygon and circle
annotation for scientific arenas (v1.0).
https://github.com/zerotonin/arena_annotator
```

### BibTeX

```bibtex
@software{geurten2026arena,
  author       = {Geurten, Bart R.H.},
  title        = {{Arena Annotator: Interactive polygon and circle
                   annotation for scientific arenas}},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/zerotonin/arena_annotator},
  license      = {MIT}
}
```
