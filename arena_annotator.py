#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀█░█▀▄░█▀▀░█▀█░█▀█░░░█▀█░█▀█░█▀█░█▀█░▀█▀░█▀█░▀█▀░█▀█░█▀▄░░         ║
 ║  ░█▀█░█▀▄░█▀▀░█░█░█▀█░░░█▀█░█░█░█░█░█░█░░█░░█▀█░░█░░█░█░█▀▄░░         ║
 ║  ░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀░▀░░         ║
 ║                                                                       ║
 ║   Interactive polygon annotation for scientific arenas    v1.0        ║
 ║   ── click. drag. export. science. ──                                 ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Mark arena boundaries or ROIs in experimental images by clicking polygon
vertices. Drag to reposition, right-click to delete. Exports to COCO JSON,
YOLO-v8 polygon, and Pascal VOC XML.

Only needs: matplotlib, numpy, Pillow — all pip/conda friendly, no Qt, no
OpenCV, no exotic deps. Runs on macOS, Linux, Windows.

    pip install matplotlib numpy Pillow

License: MIT
Author: Bart R.H. Geurten
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CONSTANTS                                   « system defaults »    │
# └─────────────────────────────────────────────────────────────────────┘
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PICK_RADIUS_PX = 15
VERTEX_SIZE = 80
VERTEX_COLOUR = "red"
EDGE_COLOUR = "orange"
FILL_COLOUR = "orange"
FILL_ALPHA = 0.40
LABEL_FONTSIZE = 9

matplotlib.rcParams["toolbar"] = "None"


# ┌─────────────────────────────────────────────────────────────────────┐
# │  IMAGE DISCOVERY                            « scanning the grid »   │
# └─────────────────────────────────────────────────────────────────────┘
def discover_images(directory=None, image_path=None, filelist=None):
    """Locate image files from one of three input modes.

    Exactly one of the three arguments should be set (enforced by CLI).

    Args:
        directory:  Scan this folder for common image extensions.
        image_path: Single image file path.
        filelist:   Text file with one image path per line (# = comment).

    Returns:
        Sorted list of resolved, absolute image paths.

    Raises:
        SystemExit: If the source is invalid or yields zero images.
    """
    paths = []

    if directory:
        d = Path(directory)
        if not d.is_dir():
            _die(f"Directory not found: {directory}")
        for ext in IMAGE_EXTENSIONS:
            paths.extend(d.glob(f"*{ext}"))
            paths.extend(d.glob(f"*{ext.upper()}"))
        paths = sorted(set(paths))

    elif image_path:
        p = Path(image_path)
        if not p.is_file():
            _die(f"Image not found: {image_path}")
        paths = [p]

    elif filelist:
        fl = Path(filelist)
        if not fl.is_file():
            _die(f"File list not found: {filelist}")
        with open(fl) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    p = Path(line)
                    if p.is_file():
                        paths.append(p)
                    else:
                        warnings.warn(f"Skipping missing image: {line}")

    if not paths:
        _die("No images found.")

    return [str(p.resolve()) for p in sorted(set(paths))]


# ┌─────────────────────────────────────────────────────────────────────┐
# │  GEOMETRY                                    « crunching pixels »   │
# └─────────────────────────────────────────────────────────────────────┘
def _polygon_area(vertices):
    """Compute polygon area via the shoelace formula.

    Works for any simple (non-self-intersecting) polygon.
    Returns 0.0 for fewer than 3 vertices.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += vertices[i][0] * vertices[j][1]
        a -= vertices[j][0] * vertices[i][1]
    return abs(a) / 2.0


# ┌─────────────────────────────────────────────────────────────────────┐
# │  EXPORT: COCO JSON                           « uploading intel »    │
# └─────────────────────────────────────────────────────────────────────┘
def export_coco(annotations, labels, output_dir):
    """Write all annotations as a single COCO-format JSON.

    Produces ``annotations_coco.json`` containing images, annotations with
    polygon segmentation masks, bounding boxes, and vertex labels stored
    under the ``attributes`` key.

    Returns:
        Path to the written JSON file.
    """
    coco = {
        "info": {
            "description": "Arena polygon annotations (arena_annotator)",
            "version": "1.0",
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "arena", "supercategory": "region"}],
    }
    ann_id = 1
    for img_id, (img_path, ann) in enumerate(annotations.items(), start=1):
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": ann["image_width"],
            "height": ann["image_height"],
        })
        verts = ann.get("vertices", [])
        if not verts:
            continue

        seg = []
        for x, y in verts:
            seg.extend([round(x, 2), round(y, 2)])

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": [seg],
            "bbox": [round(b, 2) for b in bbox],
            "area": round(_polygon_area(verts), 2),
            "iscrowd": 0,
            "attributes": {"vertex_labels": labels[: len(verts)]},
        })
        ann_id += 1

    out = os.path.join(output_dir, "annotations_coco.json")
    with open(out, "w") as fh:
        json.dump(coco, fh, indent=2)
    return out


# ┌─────────────────────────────────────────────────────────────────────┐
# │  EXPORT: YOLO POLYGON                 « normalized for the net »    │
# └─────────────────────────────────────────────────────────────────────┘
def export_yolo(annotations, output_dir):
    """Write one YOLO-v8 polygon label file per annotated image.

    Format: ``<class_id> x1 y1 x2 y2 ... xN yN`` with coordinates
    normalised to [0, 1] by image width/height. Class is always 0.

    Returns:
        List of written file paths.
    """
    written = []
    for img_path, ann in annotations.items():
        verts = ann.get("vertices", [])
        if not verts:
            continue
        w, h = ann["image_width"], ann["image_height"]
        coords = []
        for x, y in verts:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        line = "0 " + " ".join(coords)

        out = os.path.join(output_dir, Path(img_path).stem + ".txt")
        with open(out, "w") as fh:
            fh.write(line + "\n")
        written.append(out)
    return written


# ┌─────────────────────────────────────────────────────────────────────┐
# │  EXPORT: PASCAL VOC XML                    « old school markup »    │
# └─────────────────────────────────────────────────────────────────────┘
def export_voc(annotations, labels, output_dir):
    """Write one Pascal VOC XML file per annotated image.

    Standard VOC ``<bndbox>`` is included for compatibility. An additional
    ``<polygon>`` element carries the per-vertex (x, y, label) triples
    for tools that support the extended format.

    Returns:
        List of written file paths.
    """
    written = []
    for img_path, ann in annotations.items():
        verts = ann.get("vertices", [])
        if not verts:
            continue

        root = Element("annotation")
        SubElement(root, "folder").text = os.path.basename(
            os.path.dirname(img_path)
        )
        SubElement(root, "filename").text = os.path.basename(img_path)

        size_el = SubElement(root, "size")
        SubElement(size_el, "width").text = str(ann["image_width"])
        SubElement(size_el, "height").text = str(ann["image_height"])
        SubElement(size_el, "depth").text = "3"

        obj_el = SubElement(root, "object")
        SubElement(obj_el, "name").text = "arena"

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        bndbox = SubElement(obj_el, "bndbox")
        SubElement(bndbox, "xmin").text = str(int(min(xs)))
        SubElement(bndbox, "ymin").text = str(int(min(ys)))
        SubElement(bndbox, "xmax").text = str(int(max(xs)))
        SubElement(bndbox, "ymax").text = str(int(max(ys)))

        polygon_el = SubElement(obj_el, "polygon")
        for i, (x, y) in enumerate(verts):
            pt = SubElement(polygon_el, "point")
            SubElement(pt, "x").text = str(round(x, 2))
            SubElement(pt, "y").text = str(round(y, 2))
            if i < len(labels):
                SubElement(pt, "label").text = labels[i]

        xml_bytes = tostring(root, encoding="unicode")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
        pretty = "\n".join(pretty.splitlines()[1:])

        out = os.path.join(output_dir, Path(img_path).stem + ".xml")
        with open(out, "w") as fh:
            fh.write(pretty)
        written.append(out)
    return written


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ANNOTATOR GUI                                      « jacking in »  │
# └─────────────────────────────────────────────────────────────────────┘
class PolygonAnnotator:
    """Interactive matplotlib GUI for placing polygon vertices on images.

    Boots a figure window, wires up mouse/key events, and manages the
    full annotation lifecycle: click-to-place, drag-to-adjust,
    right-click-to-delete, sidecar persistence, and batch export.

    The constructor blocks on ``plt.show()`` — control returns when the
    user closes the window or presses Q/Esc.
    """

    def __init__(self, images, n_vertices, labels, output_dir, formats):
        self.images = images
        self.n_vertices = n_vertices
        self.labels = labels
        self.output_dir = output_dir
        self.formats = formats

        self.current_idx = 0
        self.annotations = {}

        self.show_labels = True
        self.show_fill = True
        self.show_help = False

        self._dragging = False
        self._drag_idx = None
        self._exported = False
        self._confirm_reset = False  # awaiting Y/N after X key

        self._load_all_sidecars()

        self.fig, self.ax = plt.subplots(1, 1, figsize=(13, 8))
        self.fig.canvas.manager.set_window_title("Arena Annotator")
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._render()
        plt.show()

    # ── sidecar I/O « crash-proof persistence » ──────────────────────
    def _sidecar_path(self, img_path):
        """Return the path for this image's JSON sidecar file."""
        return os.path.join(
            self.output_dir, Path(img_path).stem + "_annotation.json"
        )

    def _load_sidecar(self, img_path):
        """Restore vertices from a JSON sidecar if one exists on disk."""
        sp = self._sidecar_path(img_path)
        if os.path.exists(sp):
            try:
                with open(sp) as fh:
                    data = json.load(fh)
                self.annotations[img_path] = {
                    "vertices": [tuple(v) for v in data.get("vertices", [])],
                    "image_width": data.get("image_width", 0),
                    "image_height": data.get("image_height", 0),
                }
            except (json.JSONDecodeError, KeyError) as exc:
                warnings.warn(f"Corrupt sidecar {sp}, ignoring: {exc}")

    def _load_all_sidecars(self):
        """Pre-load sidecars for every image — enables session resume."""
        for img in self.images:
            self._load_sidecar(img)

    def _save_sidecar(self, img_path):
        """Persist current vertices for *img_path* to its JSON sidecar."""
        ann = self.annotations.get(img_path)
        if ann is None:
            return
        data = {
            "image_path": img_path,
            "image_width": ann["image_width"],
            "image_height": ann["image_height"],
            "vertices": [list(v) for v in ann["vertices"]],
            "labels": self.labels[: len(ann["vertices"])],
        }
        with open(self._sidecar_path(img_path), "w") as fh:
            json.dump(data, fh, indent=2)

    # ── annotation bookkeeping « state management » ──────────────────
    def _ensure_annotation(self, img_path, w, h):
        """Lazily create the annotation dict for an image if absent."""
        if img_path not in self.annotations:
            self.annotations[img_path] = {
                "vertices": [],
                "image_width": w,
                "image_height": h,
            }
        else:
            self.annotations[img_path]["image_width"] = w
            self.annotations[img_path]["image_height"] = h

    @property
    def _path(self):
        """Absolute path of the currently displayed image."""
        return self.images[self.current_idx]

    @property
    def _ann(self):
        """Annotation dict for the current image, or None."""
        return self.annotations.get(self._path)

    # ── rendering « painting the screen » ────────────────────────────
    def _render(self):
        """Redraw the full scene: image, vertices, edges, polygon, overlays."""
        self.ax.clear()

        try:
            pil_img = Image.open(self._path)
            img_arr = np.array(pil_img)
        except Exception as exc:
            self.ax.text(
                0.5, 0.5, f"Failed to load image:\n{exc}",
                transform=self.ax.transAxes, ha="center", va="center",
                color="red", fontsize=14,
            )
            self.ax.set_axis_off()
            self.fig.canvas.draw_idle()
            return

        img_w, img_h = pil_img.size
        self._ensure_annotation(self._path, img_w, img_h)
        ann = self._ann

        self.ax.imshow(img_arr, aspect="equal")
        self.ax.set_xlim(0, img_w)
        self.ax.set_ylim(img_h, 0)

        verts = ann["vertices"]
        n = len(verts)

        # Polygon / edges
        if n == self.n_vertices:
            if self.show_fill:
                poly = MplPolygon(
                    verts, closed=True,
                    facecolor=FILL_COLOUR, edgecolor=EDGE_COLOUR,
                    alpha=FILL_ALPHA, linewidth=2,
                )
                self.ax.add_patch(poly)
            else:
                xs = [v[0] for v in verts] + [verts[0][0]]
                ys = [v[1] for v in verts] + [verts[0][1]]
                self.ax.plot(xs, ys, "-", color=EDGE_COLOUR, linewidth=2, alpha=0.8)
        elif n >= 2:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            self.ax.plot(xs, ys, "-", color=EDGE_COLOUR, linewidth=2, alpha=0.7)

        # Vertices
        if n > 0:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            self.ax.scatter(
                xs, ys, s=VERTEX_SIZE, c=VERTEX_COLOUR,
                edgecolors="white", linewidth=1.5, zorder=5,
            )

            if self.show_labels:
                for i, (x, y) in enumerate(verts):
                    lbl = self.labels[i] if i < len(self.labels) else f"P{i+1}"
                    # Smart placement: flip offset when near edges
                    margin = 0.15  # fraction of image dimension
                    ox = -10 if x > img_w * (1 - margin) else 10
                    oy = 10 if y < img_h * margin else -10
                    ha = "right" if ox < 0 else "left"
                    self.ax.annotate(
                        lbl, (x, y),
                        xytext=(ox, oy), textcoords="offset points",
                        fontsize=LABEL_FONTSIZE, fontweight="bold",
                        color="white", ha=ha,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="black", alpha=0.75,
                        ),
                        zorder=6,
                    )

        if self.show_help:
            self._render_help(img_w, img_h)

        # Reset confirmation banner
        if self._confirm_reset:
            self.ax.text(
                0.5, 0.5,
                "Delete all vertices in this image?\n\n"
                "Press  (Y)es  to confirm  /  any other key to cancel",
                transform=self.ax.transAxes,
                fontsize=14, fontfamily="monospace",
                ha="center", va="center", color="white", zorder=20,
                bbox=dict(
                    boxstyle="round,pad=1.0",
                    facecolor="red", alpha=0.85,
                    edgecolor="white", linewidth=2,
                ),
            )

        # Title / status bar
        status = (
            f"[{self.current_idx + 1}/{len(self.images)}]  "
            f"{os.path.basename(self._path)}  |  "
            f"Vertices: {n}/{self.n_vertices}"
        )
        if n == self.n_vertices:
            status += "  COMPLETE"
        self.ax.set_title(status, fontsize=10, loc="left", pad=6)
        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def _render_help(self, img_w, img_h):
        """Draw the translucent help overlay with keybindings and file info."""
        text = (
            "=== ARENA ANNOTATOR HELP ===\n"
            "\n"
            "MOUSE\n"
            "  Left click          Add vertex\n"
            "  Left drag vertex    Move vertex\n"
            "  Right click vertex  Remove vertex\n"
            "\n"
            "KEYS\n"
            "  Left/Right   Previous / Next image\n"
            "  R            Repeat vertices from last image\n"
            "  X            Reset vertices (with confirmation)\n"
            "  L            Toggle labels\n"
            "  F            Toggle polygon fill\n"
            "  H            Toggle this help\n"
            "  S            Save & export now\n"
            "  Q / Esc      Save & quit\n"
            "\n"
            f"Image:   {os.path.basename(self._path)}\n"
            f"Size:    {img_w} x {img_h} px\n"
            f"Output:  {self.output_dir}\n"
            f"Formats: {', '.join(self.formats)}"
        )
        self.ax.text(
            0.02, 0.98, text,
            transform=self.ax.transAxes,
            fontsize=10, fontfamily="monospace",
            verticalalignment="top", color="white", zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="black", alpha=0.88,
                edgecolor=EDGE_COLOUR, linewidth=1.5,
            ),
        )

    # ── hit-testing « who did you click on? » ─────────────────────────
    def _find_nearby(self, event):
        """Return the index of the vertex closest to event, or None."""
        if event.xdata is None or self._ann is None:
            return None
        for i, (vx, vy) in enumerate(self._ann["vertices"]):
            disp_v = self.ax.transData.transform((vx, vy))
            dist = np.hypot(disp_v[0] - event.x, disp_v[1] - event.y)
            if dist < PICK_RADIUS_PX:
                return i
        return None

    # ── event handlers « reacting to the user » ──────────────────────
    def _on_press(self, event):
        """Handle mouse button down: add/drag vertex (LMB), delete (RMB)."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        ann = self._ann
        if ann is None:
            return

        if event.button == 1:  # left
            idx = self._find_nearby(event)
            if idx is not None:
                self._dragging = True
                self._drag_idx = idx
            elif len(ann["vertices"]) < self.n_vertices:
                x = float(np.clip(event.xdata, 0, ann["image_width"]))
                y = float(np.clip(event.ydata, 0, ann["image_height"]))
                ann["vertices"].append((x, y))
                self._save_sidecar(self._path)
                self._render()

        elif event.button == 3:  # right
            idx = self._find_nearby(event)
            if idx is not None:
                ann["vertices"].pop(idx)
                self._save_sidecar(self._path)
                self._render()

    def _on_release(self, event):
        """Finalise a drag operation — snap vertex and persist."""
        if self._dragging:
            self._dragging = False
            self._drag_idx = None
            self._save_sidecar(self._path)
            self._render()

    def _on_motion(self, event):
        """Live-update vertex position while dragging (clamped to image)."""
        if (
            self._dragging
            and event.inaxes == self.ax
            and event.xdata is not None
        ):
            ann = self._ann
            if ann and self._drag_idx is not None:
                x = float(np.clip(event.xdata, 0, ann["image_width"]))
                y = float(np.clip(event.ydata, 0, ann["image_height"]))
                ann["vertices"][self._drag_idx] = (x, y)
                self._render()

    def _on_key(self, event):
        """Central key dispatcher — see H overlay for the full binding map."""
        key = event.key

        # --- Confirmation sub-state for reset -------------------------
        if self._confirm_reset:
            if key == "y":
                ann = self._ann
                if ann:
                    ann["vertices"] = []
                    self._save_sidecar(self._path)
                self._confirm_reset = False
                self._render()
            else:
                # Any other key cancels
                self._confirm_reset = False
                self._render()
            return

        if key == "right":
            self._save_current()
            if self.current_idx < len(self.images) - 1:
                self.current_idx += 1
                self._render()

        elif key == "left":
            self._save_current()
            if self.current_idx > 0:
                self.current_idx -= 1
                self._render()

        elif key == "l":
            self.show_labels = not self.show_labels
            self._render()

        elif key == "f":
            self.show_fill = not self.show_fill
            self._render()

        elif key == "h":
            self.show_help = not self.show_help
            self._render()

        elif key == "r":
            self._repeat_previous()

        elif key == "x":
            ann = self._ann
            if ann and ann["vertices"]:
                self._confirm_reset = True
                self._render()

        elif key == "s":
            self._save_current()
            self._export_all()
            print(f"[arena_annotator] Saved & exported to {self.output_dir}")

        elif key in ("q", "escape"):
            self._save_current()
            self._export_all()
            plt.close(self.fig)

    def _on_close(self, _event):
        """Window close handler — save state and export before shutdown."""
        self._save_current()
        self._export_all()

    def _repeat_previous(self):
        """Copy vertices from the nearest preceding annotated image.

        Walks backward from current_idx.  Silent no-op if nothing found
        — handy when the arena barely moves between frames.
        """
        # Walk backwards from current_idx to find the nearest annotated image
        for i in range(self.current_idx - 1, -1, -1):
            prev = self.annotations.get(self.images[i])
            if prev and prev["vertices"]:
                ann = self._ann
                if ann is not None:
                    ann["vertices"] = list(prev["vertices"])
                    self._save_sidecar(self._path)
                    self._render()
                return
        # Nothing found — silently ignore

    # ── saving / exporting « writing to disk » ────────────────────────
    def _save_current(self):
        """Flush the active image's annotation to its sidecar."""
        self._save_sidecar(self._path)

    def _export_all(self):
        """Run all selected exporters once, then set the guard flag."""
        if self._exported:
            return
        self._exported = True

        if "coco" in self.formats:
            p = export_coco(self.annotations, self.labels, self.output_dir)
            print(f"  COCO JSON -> {p}")
        if "yolo" in self.formats:
            d = os.path.join(self.output_dir, "yolo_labels")
            os.makedirs(d, exist_ok=True)
            files = export_yolo(self.annotations, d)
            print(f"  YOLO      -> {d}/ ({len(files)} files)")
        if "voc" in self.formats:
            d = os.path.join(self.output_dir, "voc_annotations")
            os.makedirs(d, exist_ok=True)
            files = export_voc(self.annotations, self.labels, d)
            print(f"  VOC XML   -> {d}/ ({len(files)} files)")

        print("[arena_annotator] Export complete.\n")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CLI                                   « parsing the command line » │
# └─────────────────────────────────────────────────────────────────────┘
def _die(msg):
    """Print error to stderr and exit with code 1. Game over."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    """Build the argparse parser, validate inputs, resolve the output dir.

    Returns:
        (args, labels, formats, output_dir) — everything main() needs
        to boot the annotator.
    """
    parser = argparse.ArgumentParser(
        prog="arena_annotator",
        description=(
            "Arena Annotator -- interactive polygon annotation for "
            "scientific images.\n\n"
            "Place polygon vertices on arena boundaries or regions of "
            "interest.\nAnnotations are persisted as sidecar JSON files "
            "(resumable) and\nexported in COCO, YOLO, or Pascal VOC format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  %(prog)s -d ./images/ -v 4 -l "TL,TR,BR,BL"\n'
            '  %(prog)s -i photo.png -v 6 -l "p1,p2,p3,p4,p5,p6" -a yolo,coco\n'
            "  %(prog)s -f filelist.txt -v 4 -o ./output/ -a voc\n"
            "\n"
            "key bindings (inside the annotation window):\n"
            "  Left click           Add vertex (up to -v count)\n"
            "  Left drag on vertex  Move vertex\n"
            "  Right click vertex   Delete vertex\n"
            "  Left / Right arrow   Previous / Next image\n"
            "  R                    Repeat vertices from previous image\n"
            "  X                    Reset all vertices (with confirmation)\n"
            "  L                    Toggle vertex labels\n"
            "  F                    Toggle polygon fill\n"
            "  H                    Toggle help overlay\n"
            "  S                    Save & export\n"
            "  Q / Esc              Save & quit\n"
        ),
    )

    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "-d", "--directory",
        help="Directory of images to annotate.",
    )
    inp.add_argument(
        "-i", "--image",
        help="Single image file to annotate.",
    )
    inp.add_argument(
        "-f", "--filelist",
        help="Text file listing image paths (one per line, # comments ok).",
    )

    parser.add_argument(
        "-v", "--vertices",
        type=int, required=True,
        help="Number of polygon vertices (>= 3).",
    )
    parser.add_argument(
        "-l", "--labels",
        type=str, default=None,
        help=(
            'Comma-separated vertex labels '
            '(e.g. "top_left,top_right,bottom_right,bottom_left"). '
            "Count must match -v. Default: P1, P2, ..."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=str, default=None,
        help="Output directory for annotations (default: image source dir).",
    )
    parser.add_argument(
        "-a", "--formats",
        type=str, default="coco",
        help=(
            "Export format(s), comma-separated. "
            "Choices: coco, yolo, voc (default: coco)."
        ),
    )

    args = parser.parse_args()

    if args.vertices < 3:
        parser.error("--vertices must be >= 3 to form a polygon.")

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
        if len(labels) != args.vertices:
            parser.error(
                f"Label count ({len(labels)}) does not match "
                f"vertex count ({args.vertices})."
            )
    else:
        labels = [f"P{i + 1}" for i in range(args.vertices)]

    valid_fmts = {"coco", "yolo", "voc"}
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    for fmt in formats:
        if fmt not in valid_fmts:
            parser.error(
                f"Unknown format '{fmt}'. Choose from: {', '.join(sorted(valid_fmts))}"
            )

    if args.output:
        output_dir = args.output
    elif args.directory:
        output_dir = args.directory
    elif args.image:
        output_dir = str(Path(args.image).resolve().parent)
    else:
        output_dir = str(Path(args.filelist).resolve().parent)

    os.makedirs(output_dir, exist_ok=True)

    return args, labels, formats, output_dir


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ENTRY POINT                                  « start sequence »    │
# └─────────────────────────────────────────────────────────────────────┘
def main():
    """Parse CLI, discover images, print status, and launch the GUI."""
    args, labels, formats, output_dir = parse_args()

    images = discover_images(
        directory=args.directory,
        image_path=args.image,
        filelist=args.filelist,
    )

    print("┌──────────────────────────────────────┐")
    print("│  ARENA ANNOTATOR v1.0                │")
    print("│  « click. drag. export. science. »   │")
    print("└──────────────────────────────────────┘")
    print(f"  Images:   {len(images)}")
    print(f"  Vertices: {args.vertices}")
    print(f"  Labels:   {', '.join(labels)}")
    print(f"  Output:   {output_dir}")
    print(f"  Formats:  {', '.join(formats)}")
    print("  Press H inside the window for help.\n")

    PolygonAnnotator(images, args.vertices, labels, output_dir, formats)


if __name__ == "__main__":
    main()
