```
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀█░█▀▄░█▀▀░█▀█░█▀█░░░█▀█░█▀█░█▀█░█▀█░▀█▀░█▀█░▀█▀░█▀█░█▀▄░░         ║
 ║  ░█▀█░█▀▄░█▀▀░█░█░█▀█░░░█▀█░█░█░█░█░█░█░░█░░█▀█░░█░░█░█░█▀▄░░         ║
 ║  ░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀░▀░░         ║
 ║                                                                       ║
 ║   Interactive polygon annotation for scientific arenas    v1.0        ║
 ║   ── click. drag. export. science. ──                                 ║
 ╚═══════════════════════════════════════════════════════════════════════╝
```

# Arena Annotator

A single-file, dependency-light tool for annotating polygon regions in experimental images. Built for the common workflow in behavioural neuroscience where you need to map arena boundaries from camera frames to real-world coordinates.

Click vertices onto your arena, drag to adjust, export to standard formats. Sessions are crash-proof and resumable. That's it.

## Motivation

If you track animals in arenas — open fields, mazes, thermal gradient setups — you eventually need to define the arena boundary in pixel coordinates so you can project tracked positions into millimetres. This usually means writing a quick script with `cv2.setMouseCallback` for the twentieth time, losing the annotations when the script crashes, and then doing it again for 200 images.

Arena Annotator replaces that loop. It persists every click to disk immediately, resumes where you left off, lets you propagate vertices across frames with one keypress, and exports to the formats your downstream pipeline already expects.

## When to use this — and when not to

This tool does one thing: mark a fixed polygon on a batch of images. If that's what you need, it will take you 30 seconds to install and less to learn.

If you need more — multi-class labelling, freehand masks, bounding boxes, team workflows, model-in-the-loop pre-annotation — use one of the full-featured annotation platforms:

| Tool | Scope | Notes |
|------|-------|-------|
| [CVAT](https://github.com/cvat-ai/cvat) | Image & video annotation | Open source (Intel). Interpolation, tracking, SAM integration. The current standard for serious CV annotation. Docker-based. |
| [Label Studio](https://github.com/HumanSignal/label-studio) | Multi-modal (image, text, audio, video) | Open source (HumanSignal). Very flexible config system. Good if you label more than just images. |
| [LabelImg](https://github.com/HumanSignal/labelImg) | Lightweight image annotation | The classic. Archived since Feb 2024, now folded into Label Studio. Still works for simple bounding-box tasks. |
| [LabelMe](https://github.com/labelmeai/labelme) | Polygonal image annotation | Open source. Closest in spirit to Arena Annotator but more general-purpose. Good if you need freehand polygons and multiple object classes. |
| [Roboflow](https://roboflow.com/) | End-to-end CV pipeline | Commercial, free tier. Annotation + augmentation + training + deployment in one platform. |

Arena Annotator fills the gap below all of these: when you just need a polygon on each frame, you don't want Docker, and you don't want to register for anything.

## Installation

Three dependencies, all in every major conda channel and on PyPI:

```bash
pip install matplotlib numpy Pillow
```

Or with conda:

```bash
conda install matplotlib numpy pillow
```

Then just put `arena_annotator.py` somewhere on your `$PATH`, or call it directly:

```bash
python arena_annotator.py --help
```

No Qt bindings, no OpenCV, no compiled extensions. Works on macOS, Linux, and Windows.

## Quick start

Annotate a rectangular arena across a directory of frames:

```bash
python arena_annotator.py \
    -d ./trial_images/ \
    -v 4 \
    -l "TL,TR,BR,BL" \
    -a coco,yolo
```

This opens a matplotlib window. Left-click to place the four vertices in order, drag to adjust. Press → to advance to the next image. Press Q when done — COCO JSON and YOLO label files appear in `./trial_images/`.

## Usage

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

### Examples

```bash
# Hexagonal arena, single image, all three export formats
python arena_annotator.py -i frame.png -v 6 -l "p1,p2,p3,p4,p5,p6" -a coco,yolo,voc

# From a file list, output to a separate directory
python arena_annotator.py -f paths.txt -v 4 -o ./annotations/

# Minimal — just a triangle, default labels, COCO only
python arena_annotator.py -d ./imgs/ -v 3
```

## Keybindings

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
| **H** | Help overlay (also shows current file path and output directory) |
| **S** | Save & export now |
| **Q** / Esc | Save & quit |

## Export formats

### COCO JSON

A single `annotations_coco.json` covering all images. Standard COCO structure with polygon segmentation, bounding box, area, and an `attributes.vertex_labels` field carrying your label names.

```json
{
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "segmentation": [[102.5, 48.3, 537.8, 51.1, 534.2, 389.7, 98.9, 386.5]],
    "bbox": [98.9, 48.3, 438.9, 341.4],
    "area": 147316.32,
    "attributes": {"vertex_labels": ["TL", "TR", "BR", "BL"]}
  }]
}
```

### YOLO v8 polygon

One `.txt` per image in `yolo_labels/`. Coordinates normalised to `[0, 1]` by image dimensions. Class is always `0`.

```
0 0.160156 0.100625 0.840312 0.106458 0.834688 0.811875 0.154531 0.805208
```

### Pascal VOC XML

One `.xml` per image in `voc_annotations/`. Standard `<bndbox>` for compatibility plus a `<polygon>` element with per-vertex coordinates and labels.

```xml
<annotation>
  <filename>frame_0042.png</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>arena</name>
    <bndbox>
      <xmin>98</xmin><ymin>48</ymin><xmax>537</xmax><ymax>389</ymax>
    </bndbox>
    <polygon>
      <point><x>102.5</x><y>48.3</y><label>TL</label></point>
      <point><x>537.8</x><y>51.1</y><label>TR</label></point>
      ...
    </polygon>
  </object>
</annotation>
```

## Session persistence

Every vertex change is instantly written to a JSON sidecar file (`<image_stem>_annotation.json`) in the output directory. If the process crashes, the window is closed accidentally, or you come back the next day, the annotator picks up exactly where you left off.

```json
{
  "image_path": "/data/experiment_01/frame_0042.png",
  "image_width": 640,
  "image_height": 480,
  "vertices": [[102.5, 48.3], [537.8, 51.1], [534.2, 389.7], [98.9, 386.5]],
  "labels": ["TL", "TR", "BR", "BL"]
}
```

These sidecars are also a convenient intermediate format if your downstream code just wants to read the raw pixel coordinates directly.

## Design notes

**Single file.** No package structure, no `setup.py`, no build step. Copy the script, install three packages, go.

**matplotlib backend.** Chosen deliberately over OpenCV `highgui` or Qt for maximum portability. The only trade-off is that rendering during drag is not buttery smooth on very large images — but it's fine for the 640×480 to 2048×2048 range typical in behavioural setups.

**Smart label placement.** Labels flip to the opposite side of the vertex when it's near an image edge, so annotations at arena borders remain readable.

**Repeat key (R).** In longitudinal experiments with a fixed camera, the arena barely moves between sessions. Press R to copy the polygon from the last annotated frame and adjust from there — typically saves 90% of the clicking.

## Author

**Bart R.H. Geurten**
Department of Zoology, University of Otago, Dunedin, New Zealand

- [University profile](https://www.otago.ac.nz/zoology/staff/dr-bart-geurten)
- [Google Scholar](https://scholar.google.de/citations?user=OAm7kgcAAAAJ&hl=en)

## License

[MIT](LICENSE)

## Citation

If this tool is useful in your published work, a citation or acknowledgement is appreciated:

```
Geurten, B. R. H. (2026). Arena Annotator: Interactive polygon annotation
for scientific arenas (v1.0). https://github.com/zerotonin/arena_annotator
```
