#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀▀░▀█▀░█▀▄░█▀▀░█░░░█▀▀░░░█▀█░█▀█░█▀█░█▀█░▀█▀░█▀█░▀█▀░█▀█░█▀▄░  ║
 ║  ░█░░░░█░░█▀▄░█░░░█░░░█▀▀░░░█▀█░█░█░█░█░█░█░░█░░█▀█░░█░░█░█░█▀▄░  ║
 ║  ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀░▀░  ║
 ║                                                                       ║
 ║   Interactive circle annotation for round scientific arenas  v1.0     ║
 ║   ── click. drag. export. science. ──                                 ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Mark circular arena boundaries in experimental images by clicking a centre
point and then a second point to set the radius. Drag centre to reposition,
drag the rim handle to resize. Exports to COCO JSON, YOLO-v8, and Pascal
VOC XML.

Only needs: matplotlib, numpy, Pillow — all pip/conda friendly, no Qt, no
OpenCV, no exotic deps. Runs on macOS, Linux, Windows.

    pip install matplotlib numpy Pillow

License: MIT
Author: Bart R.H. Geurten
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle
from PIL import Image

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CONSTANTS                                   « system defaults »    │
# └─────────────────────────────────────────────────────────────────────┘
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PICK_RADIUS_PX = 15
CENTRE_SIZE = 100
RIM_HANDLE_SIZE = 60
CENTRE_COLOUR = "red"
RIM_COLOUR = "cyan"
EDGE_COLOUR = "orange"
FILL_COLOUR = "orange"
FILL_ALPHA = 0.25
LABEL_FONTSIZE = 9
N_POLYGON_VERTICES = 64  # vertices when approximating circle as polygon for export

try:
    matplotlib.rcParams["toolbar"] = "None"
except TypeError:
    pass  # mocked matplotlib during doc builds


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
def _circle_area(radius):
    """Compute area of a circle from its radius.

    Args:
        radius: Circle radius in pixels.

    Returns:
        Area in square pixels. Returns 0.0 if radius is non-positive.
    """
    if radius <= 0:
        return 0.0
    return math.pi * radius * radius


def _circle_to_polygon(cx, cy, radius, n_vertices=N_POLYGON_VERTICES):
    """Approximate a circle as an n-sided regular polygon.

    Useful for exporting to annotation formats that only support
    polygonal segmentation masks (COCO, YOLO, VOC).

    Args:
        cx: Centre x coordinate in pixels.
        cy: Centre y coordinate in pixels.
        radius: Circle radius in pixels.
        n_vertices: Number of polygon vertices for the approximation.

    Returns:
        List of (x, y) tuples forming the polygon vertices, equally
        spaced around the circumference starting at angle 0.
    """
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]


# ┌─────────────────────────────────────────────────────────────────────┐
# │  EXPORT: COCO JSON                           « uploading intel »    │
# └─────────────────────────────────────────────────────────────────────┘
def export_coco(annotations, output_dir):
    """Write all annotations as a single COCO-format JSON.

    Produces ``annotations_coco.json`` containing images and annotations
    with polygon segmentation masks (circle approximated as N-gon),
    bounding boxes, and the native circle parameters (centre_x, centre_y,
    radius) stored under the ``attributes`` key.

    Args:
        annotations: Dict mapping image paths to annotation dicts, each
            containing ``centre``, ``radius``, ``image_width``, and
            ``image_height`` keys.
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the written JSON file.
    """
    coco = {
        "info": {
            "description": "Arena circle annotations (circle_annotator)",
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
        centre = ann.get("centre")
        radius = ann.get("radius")
        if centre is None or radius is None or radius <= 0:
            continue

        cx, cy = centre
        verts = _circle_to_polygon(cx, cy, radius)

        seg = []
        for x, y in verts:
            seg.extend([round(x, 2), round(y, 2)])

        bbox = [
            round(cx - radius, 2),
            round(cy - radius, 2),
            round(2 * radius, 2),
            round(2 * radius, 2),
        ]

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": [seg],
            "bbox": bbox,
            "area": round(_circle_area(radius), 2),
            "iscrowd": 0,
            "attributes": {
                "shape": "circle",
                "centre_x": round(cx, 2),
                "centre_y": round(cy, 2),
                "radius": round(radius, 2),
            },
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

    The circle is approximated as an N-gon. Format:
    ``<class_id> x1 y1 x2 y2 ... xN yN`` with coordinates normalised
    to [0, 1] by image width/height. Class is always 0.

    Args:
        annotations: Dict mapping image paths to annotation dicts.
        output_dir: Directory to write label files into.

    Returns:
        List of written file paths.
    """
    written = []
    for img_path, ann in annotations.items():
        centre = ann.get("centre")
        radius = ann.get("radius")
        if centre is None or radius is None or radius <= 0:
            continue

        w, h = ann["image_width"], ann["image_height"]
        cx, cy = centre
        verts = _circle_to_polygon(cx, cy, radius)

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
def export_voc(annotations, output_dir):
    """Write one Pascal VOC XML file per annotated image.

    Standard VOC ``<bndbox>`` is included for compatibility. An additional
    ``<circle>`` element carries the native centre and radius, and a
    ``<polygon>`` element carries the N-gon approximation for tools that
    only support polygonal annotations.

    Args:
        annotations: Dict mapping image paths to annotation dicts.
        output_dir: Directory to write XML files into.

    Returns:
        List of written file paths.
    """
    written = []
    for img_path, ann in annotations.items():
        centre = ann.get("centre")
        radius = ann.get("radius")
        if centre is None or radius is None or radius <= 0:
            continue

        cx, cy = centre

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

        bndbox = SubElement(obj_el, "bndbox")
        SubElement(bndbox, "xmin").text = str(int(cx - radius))
        SubElement(bndbox, "ymin").text = str(int(cy - radius))
        SubElement(bndbox, "xmax").text = str(int(cx + radius))
        SubElement(bndbox, "ymax").text = str(int(cy + radius))

        circle_el = SubElement(obj_el, "circle")
        SubElement(circle_el, "cx").text = str(round(cx, 2))
        SubElement(circle_el, "cy").text = str(round(cy, 2))
        SubElement(circle_el, "radius").text = str(round(radius, 2))

        polygon_el = SubElement(obj_el, "polygon")
        for x, y in _circle_to_polygon(cx, cy, radius):
            pt = SubElement(polygon_el, "point")
            SubElement(pt, "x").text = str(round(x, 2))
            SubElement(pt, "y").text = str(round(y, 2))

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
class CircleAnnotator:
    """Interactive matplotlib GUI for placing circles on images.

    Interaction model:
        1. First left-click sets the circle **centre**.
        2. Second left-click sets the **radius** (distance from centre
           to the clicked point).
        3. After both are placed, drag the centre to reposition and drag
           the rim handle (shown at 3-o'clock on the circumference) to
           resize.

    The constructor blocks on ``plt.show()`` — control returns when the
    user closes the window or presses Q/Esc.

    Args:
        images: List of absolute image file paths.
        output_dir: Directory for sidecar JSON and exported annotations.
        formats: List of export format strings (``coco``, ``yolo``, ``voc``).
    """

    def __init__(self, images, output_dir, formats, grid=False, context_dir=None):
        self.images = images
        self.output_dir = output_dir
        self.formats = formats

        self.current_idx = 0
        self.annotations: Dict[str, dict] = {}

        self.show_fill = True
        self.show_labels = True
        self.show_help = False

        self._dragging_centre = False
        self._dragging_rim = False
        self._exported = False
        self._confirm_reset = False
        self._goto_buffer = None   # None = not in go-to mode; str = digits typed

        # Optional split view: the main annotator (left) beside a 4x4 grid of
        # pre-sampled context frames (right).  The frames are supplied as images
        # (baked by an external tool) -- the annotator stays vanilla, no video/CV.
        self.grid = bool(grid)
        self.context_dir = context_dir
        self._ctx_for = None       # image the context frames were last drawn for
        self.ctx_axes = []
        self.ctx_rings = []

        self._load_all_sidecars()

        if self.grid:
            self.fig = plt.figure(figsize=(16, 8))
            gs = self.fig.add_gridspec(4, 8, wspace=0.03, hspace=0.03)
            self.ax = self.fig.add_subplot(gs[:, :4])
            self.ctx_axes = [self.fig.add_subplot(gs[r, 4 + c])
                             for r in range(4) for c in range(4)]
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(13, 8))
        self.fig.canvas.manager.set_window_title("Circle Annotator")
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
        """Return the path for this image's JSON sidecar file.

        Args:
            img_path: Absolute path to the image file.

        Returns:
            Sidecar file path as a string.
        """
        return os.path.join(
            self.output_dir, Path(img_path).stem + "_circle_annotation.json"
        )

    def _load_sidecar(self, img_path):
        """Restore circle annotation from a JSON sidecar if one exists.

        Args:
            img_path: Absolute path to the image file.
        """
        sp = self._sidecar_path(img_path)
        if os.path.exists(sp):
            try:
                with open(sp) as fh:
                    data = json.load(fh)
                centre = data.get("centre")
                radius = data.get("radius")
                self.annotations[img_path] = {
                    "centre": tuple(centre) if centre else None,
                    "radius": radius,
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
        """Persist current circle for *img_path* to its JSON sidecar.

        Args:
            img_path: Absolute path to the image file.
        """
        ann = self.annotations.get(img_path)
        if ann is None:
            return
        data = {
            "image_path": img_path,
            "image_width": ann["image_width"],
            "image_height": ann["image_height"],
            "centre": list(ann["centre"]) if ann["centre"] else None,
            "radius": ann["radius"],
        }
        with open(self._sidecar_path(img_path), "w") as fh:
            json.dump(data, fh, indent=2)

    # ── annotation bookkeeping « state management » ──────────────────
    def _ensure_annotation(self, img_path, w, h):
        """Lazily create the annotation dict for an image if absent.

        Args:
            img_path: Absolute path to the image file.
            w: Image width in pixels.
            h: Image height in pixels.
        """
        if img_path not in self.annotations:
            self.annotations[img_path] = {
                "centre": None,
                "radius": None,
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

    @property
    def _is_complete(self):
        """True when the current image has both centre and radius set."""
        ann = self._ann
        if ann is None:
            return False
        return ann["centre"] is not None and ann["radius"] is not None and ann["radius"] > 0

    def _rim_handle_xy(self):
        """Return (x, y) of the rim drag-handle on the circle.

        The handle only encodes the radius, so it can sit at any angle; it is
        kept at the angle where the user last clicked/dragged it (``rim_angle``,
        default 3-o'clock).  Because that click is clamped to the image, the
        handle stays inside the image even when the rim itself extends past the
        crop edge -- so it never lands off-canvas and needs no view resizing.

        Returns:
            Tuple (x, y), or None if the circle is incomplete.
        """
        ann = self._ann
        if ann is None or ann["centre"] is None or ann["radius"] is None:
            return None
        cx, cy = ann["centre"]
        a = ann.get("rim_angle", 0.0)
        return (cx + ann["radius"] * math.cos(a), cy + ann["radius"] * math.sin(a))

    # ── rendering « painting the screen » ────────────────────────────
    def _render(self):
        """Redraw the full scene: image, circle, handles, overlays."""
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

        centre = ann["centre"]
        radius = ann["radius"]

        if centre is not None:
            cx, cy = centre

            # Draw circle
            if radius is not None and radius > 0:
                circle_patch = MplCircle(
                    (cx, cy), radius,
                    fill=self.show_fill,
                    facecolor=FILL_COLOUR if self.show_fill else "none",
                    edgecolor=EDGE_COLOUR,
                    alpha=FILL_ALPHA if self.show_fill else 0.9,
                    linewidth=2,
                )
                self.ax.add_patch(circle_patch)

                # Rim handle at 3-o'clock
                rim_x, rim_y = self._rim_handle_xy()
                self.ax.scatter(
                    [rim_x], [rim_y], s=RIM_HANDLE_SIZE, c=RIM_COLOUR,
                    edgecolors="white", linewidth=1.5, zorder=6,
                    marker="D",
                )

                # Dashed radius line
                self.ax.plot(
                    [cx, rim_x], [cy, rim_y], "--",
                    color=RIM_COLOUR, linewidth=1, alpha=0.7, zorder=4,
                )

                if self.show_labels:
                    lbl = f"r = {radius:.1f} px"
                    mid_x = (cx + rim_x) / 2
                    mid_y = (cy + rim_y) / 2
                    self.ax.annotate(
                        lbl, (mid_x, mid_y),
                        xytext=(0, -12), textcoords="offset points",
                        fontsize=LABEL_FONTSIZE, fontweight="bold",
                        color="white", ha="center",
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="black", alpha=0.75,
                        ),
                        zorder=7,
                    )

            # Centre marker
            self.ax.scatter(
                [cx], [cy], s=CENTRE_SIZE, c=CENTRE_COLOUR,
                edgecolors="white", linewidth=1.5, zorder=5,
                marker="+",
            )
            # Crosshair
            self.ax.scatter(
                [cx], [cy], s=CENTRE_SIZE, c=CENTRE_COLOUR,
                edgecolors="white", linewidth=1.5, zorder=5,
            )

            if self.show_labels:
                lbl = f"C ({cx:.0f}, {cy:.0f})"
                margin = 0.15
                ox = -10 if cx > img_w * (1 - margin) else 10
                oy = 10 if cy < img_h * margin else -10
                ha = "right" if ox < 0 else "left"
                self.ax.annotate(
                    lbl, (cx, cy),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=LABEL_FONTSIZE, fontweight="bold",
                    color="white", ha=ha,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="black", alpha=0.75,
                    ),
                    zorder=7,
                )

        if self.show_help:
            self._render_help(img_w, img_h)

        # Reset confirmation banner
        if self._confirm_reset:
            self.ax.text(
                0.5, 0.5,
                "Delete circle annotation for this image?\n\n"
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

        # Go-to input prompt
        if self._goto_buffer is not None:
            n = len(self.images)
            self.ax.text(
                0.5, 0.5,
                f"Go to image  [1-{n}]:  {self._goto_buffer}_\n\n"
                "type digits  /  Enter to jump  /  Backspace  /  Esc to cancel",
                transform=self.ax.transAxes,
                fontsize=14, fontfamily="monospace",
                ha="center", va="center", color="white", zorder=20,
                bbox=dict(
                    boxstyle="round,pad=1.0",
                    facecolor="#0072B2", alpha=0.9,
                    edgecolor="white", linewidth=2,
                ),
            )

        # Title / status bar
        if self._is_complete:
            state = "COMPLETE"
        elif centre is not None:
            state = "click rim to set radius"
        else:
            state = "click to set centre"

        status = (
            f"[{self.current_idx + 1}/{len(self.images)}]  "
            f"{os.path.basename(self._path)}  |  "
            f"{state}"
        )
        self.ax.set_title(status, fontsize=10, loc="left", pad=6)
        self.ax.set_axis_off()
        self._render_context()
        self.fig.canvas.draw_idle()

    def _context_frames(self, image_path):
        """Sorted pre-sampled context frames for an image, or [] if none.

        Convention: ``<context_dir or <image_dir>/context>/<image_stem>/*.png``.
        The frames are baked by an external tool -- the annotator never touches
        video (stays vanilla).
        """
        if not self.grid:
            return []
        p = Path(image_path)
        base = Path(self.context_dir) if self.context_dir else p.parent / "context"
        d = base / p.stem
        return sorted(str(f) for f in d.glob("*.png")) if d.is_dir() else []

    def _render_context(self):
        """Draw the 16 context frames once per image; move the ring patches live.

        The frames are static, so they are imshow-n only when the image changes;
        every render just repositions the 16 ring patches to the current circle,
        which keeps dragging responsive.
        """
        if not self.grid or not self.ctx_axes:
            return
        frames = self._context_frames(self._path)
        if self._ctx_for != self._path:                    # new image -> redraw frames
            self._ctx_for = self._path
            self.ctx_rings = []
            for i, ax in enumerate(self.ctx_axes):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                if i < len(frames):
                    try:
                        ax.imshow(np.array(Image.open(frames[i])))
                    except Exception:                      # noqa: BLE001
                        ax.set_axis_off()
                    ring = MplCircle((0, 0), 0, fill=False, edgecolor=EDGE_COLOUR,
                                     linewidth=1.2, visible=False)
                    ax.add_patch(ring)
                    self.ctx_rings.append(ring)
                else:
                    ax.set_axis_off()
                    self.ctx_rings.append(None)
        ann = self._ann
        centre = ann["centre"] if ann else None
        radius = ann["radius"] if ann else None
        for ring in self.ctx_rings:                         # live: just move the patches
            if ring is None:
                continue
            if centre is not None and radius:
                ring.set_center((centre[0], centre[1]))
                ring.set_radius(radius)
                ring.set_visible(True)
            else:
                ring.set_visible(False)

    def _render_help(self, img_w, img_h):
        """Draw the translucent help overlay with keybindings and file info.

        Args:
            img_w: Image width in pixels (for display in the overlay).
            img_h: Image height in pixels (for display in the overlay).
        """
        text = (
            "=== CIRCLE ANNOTATOR HELP ===\n"
            "\n"
            "MOUSE\n"
            "  1st left click      Set circle centre\n"
            "  2nd left click      Set radius\n"
            "  Drag centre         Move circle\n"
            "  Drag rim handle     Resize circle\n"
            "  Right click         Remove annotation\n"
            "\n"
            "KEYS\n"
            "  Left/Right   Previous / Next image\n"
            "  U / Tab      Jump to first unlabelled image\n"
            "  G            Go to image number (type + Enter)\n"
            "  R            Repeat circle from last image\n"
            "  X            Reset circle (with confirmation)\n"
            "  L            Toggle labels\n"
            "  F            Toggle circle fill\n"
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
    def _hit_centre(self, event):
        """Check if event is near the centre marker.

        Args:
            event: Matplotlib mouse event.

        Returns:
            True if the click is within PICK_RADIUS_PX of the centre.
        """
        ann = self._ann
        if ann is None or ann["centre"] is None:
            return False
        cx, cy = ann["centre"]
        disp = self.ax.transData.transform((cx, cy))
        dist = np.hypot(disp[0] - event.x, disp[1] - event.y)
        return dist < PICK_RADIUS_PX

    def _hit_rim(self, event):
        """Check if event is near the rim handle.

        Args:
            event: Matplotlib mouse event.

        Returns:
            True if the click is within PICK_RADIUS_PX of the rim handle.
        """
        rim = self._rim_handle_xy()
        if rim is None:
            return False
        disp = self.ax.transData.transform(rim)
        dist = np.hypot(disp[0] - event.x, disp[1] - event.y)
        return dist < PICK_RADIUS_PX

    # ── event handlers « reacting to the user » ──────────────────────
    def _on_press(self, event):
        """Handle mouse button down: place/drag centre or rim, delete (RMB).

        Two-click placement model:
            - If no centre exists, left-click places the centre.
            - If centre exists but no radius, left-click sets the radius
              as the distance from centre to the click.
            - If both exist, left-click on centre or rim initiates a drag.

        Right-click anywhere resets the annotation (centre + radius cleared).

        Args:
            event: Matplotlib button_press_event.
        """
        if event.inaxes != self.ax or event.xdata is None:
            return
        ann = self._ann
        if ann is None:
            return

        if event.button == 1:  # left
            if ann["centre"] is None:
                # Place centre
                x = float(np.clip(event.xdata, 0, ann["image_width"]))
                y = float(np.clip(event.ydata, 0, ann["image_height"]))
                ann["centre"] = (x, y)
                self._save_sidecar(self._path)
                self._render()

            elif ann["radius"] is None or ann["radius"] <= 0:
                # Place radius
                cx, cy = ann["centre"]
                x = float(np.clip(event.xdata, 0, ann["image_width"]))
                y = float(np.clip(event.ydata, 0, ann["image_height"]))
                ann["radius"] = float(np.hypot(x - cx, y - cy))
                ann["rim_angle"] = math.atan2(y - cy, x - cx)  # keep handle where clicked
                self._save_sidecar(self._path)
                self._render()

            else:
                # Drag existing handles
                if self._hit_centre(event):
                    self._dragging_centre = True
                elif self._hit_rim(event):
                    self._dragging_rim = True

        elif event.button == 3:  # right — clear annotation
            if ann["centre"] is not None:
                ann["centre"] = None
                ann["radius"] = None
                self._save_sidecar(self._path)
                self._render()

    def _on_release(self, event):
        """Finalise a drag operation — snap and persist.

        Args:
            event: Matplotlib button_release_event.
        """
        if self._dragging_centre or self._dragging_rim:
            self._dragging_centre = False
            self._dragging_rim = False
            self._save_sidecar(self._path)
            self._render()

    def _on_motion(self, event):
        """Live-update centre or radius while dragging (clamped to image).

        Args:
            event: Matplotlib motion_notify_event.
        """
        if event.inaxes != self.ax or event.xdata is None:
            return
        ann = self._ann
        if ann is None:
            return

        if self._dragging_centre and ann["centre"] is not None:
            # Keep the rim marker anchored in image space: moving the centre
            # re-fits the radius/angle to the *stationary* handle rather than
            # translating the whole circle.
            handle = self._rim_handle_xy()
            x = float(np.clip(event.xdata, 0, ann["image_width"]))
            y = float(np.clip(event.ydata, 0, ann["image_height"]))
            ann["centre"] = (x, y)
            if handle is not None:
                hx, hy = handle
                ann["radius"] = float(np.hypot(hx - x, hy - y))
                ann["rim_angle"] = math.atan2(hy - y, hx - x)
            self._render()

        elif self._dragging_rim and ann["centre"] is not None:
            cx, cy = ann["centre"]
            x = float(np.clip(event.xdata, 0, ann["image_width"]))
            y = float(np.clip(event.ydata, 0, ann["image_height"]))
            ann["radius"] = float(np.hypot(x - cx, y - cy))
            ann["rim_angle"] = math.atan2(y - cy, x - cx)  # handle follows the cursor
            self._render()

    def _on_key(self, event):
        """Central key dispatcher — see H overlay for the full binding map.

        Args:
            event: Matplotlib key_press_event.
        """
        key = event.key

        # --- Confirmation sub-state for reset -------------------------
        if self._confirm_reset:
            if key == "y":
                ann = self._ann
                if ann:
                    ann["centre"] = None
                    ann["radius"] = None
                    self._save_sidecar(self._path)
                self._confirm_reset = False
                self._render()
            else:
                self._confirm_reset = False
                self._render()
            return

        # --- Go-to input sub-state (digits only; strings can't enter) ----
        if self._goto_buffer is not None:
            if key == "enter":
                self._commit_goto()
            elif key == "escape":
                self._goto_buffer = None
                self._render()
            elif key == "backspace":
                self._goto_buffer = self._goto_buffer[:-1]
                self._render()
            elif key is not None and len(key) == 1 and key.isdigit():
                if len(self._goto_buffer) < 6:
                    self._goto_buffer += key
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

        elif key in ("u", "tab"):
            self._goto_first_unlabelled()

        elif key == "g":
            self._goto_buffer = ""
            self._render()

        elif key == "r":
            self._repeat_previous()

        elif key == "x":
            ann = self._ann
            if ann and ann["centre"] is not None:
                self._confirm_reset = True
                self._render()

        elif key == "s":
            self._save_current()
            self._export_all()
            print(f"[circle_annotator] Saved & exported to {self.output_dir}")

        elif key in ("q", "escape"):
            self._save_current()
            self._export_all()
            plt.close(self.fig)

    def _on_close(self, _event):
        """Window close handler — save state and export before shutdown."""
        self._save_current()
        self._export_all()

    def _is_labelled(self, idx):
        """True when image ``idx`` has a complete circle (centre + radius)."""
        ann = self.annotations.get(self.images[idx])
        return bool(ann and ann["centre"] is not None
                    and ann["radius"] is not None and ann["radius"] > 0)

    def _goto_first_unlabelled(self):
        """Jump to the first image without a complete circle (U / Tab)."""
        for i in range(len(self.images)):
            if not self._is_labelled(i):
                self._save_current()
                self.current_idx = i
                self._render()
                return
        print("[circle_annotator] all images are labelled.")

    def _commit_goto(self):
        """Parse the go-to buffer and jump; reject 0, > max, empty, non-int."""
        buf = (self._goto_buffer or "").strip()
        self._goto_buffer = None
        try:
            num = int(buf)
        except (ValueError, TypeError):
            self._render()
            return
        if 1 <= num <= len(self.images):
            self._save_current()
            self.current_idx = num - 1
        self._render()

    def _repeat_previous(self):
        """Copy circle from the nearest preceding annotated image.

        Walks backward from current_idx. Silent no-op if nothing found
        — handy when the arena barely moves between frames.
        """
        for i in range(self.current_idx - 1, -1, -1):
            prev = self.annotations.get(self.images[i])
            if prev and prev["centre"] is not None and prev["radius"] is not None:
                ann = self._ann
                if ann is not None:
                    ann["centre"] = prev["centre"]
                    ann["radius"] = prev["radius"]
                    self._save_sidecar(self._path)
                    self._render()
                return

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
            p = export_coco(self.annotations, self.output_dir)
            print(f"  COCO JSON -> {p}")
        if "yolo" in self.formats:
            d = os.path.join(self.output_dir, "yolo_labels")
            os.makedirs(d, exist_ok=True)
            files = export_yolo(self.annotations, d)
            print(f"  YOLO      -> {d}/ ({len(files)} files)")
        if "voc" in self.formats:
            d = os.path.join(self.output_dir, "voc_annotations")
            os.makedirs(d, exist_ok=True)
            files = export_voc(self.annotations, d)
            print(f"  VOC XML   -> {d}/ ({len(files)} files)")

        print("[circle_annotator] Export complete.\n")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CLI                                   « parsing the command line » │
# └─────────────────────────────────────────────────────────────────────┘
def _die(msg):
    """Print error to stderr and exit with code 1. Game over.

    Args:
        msg: Error message string.
    """
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    """Build the argparse parser, validate inputs, resolve the output dir.

    Returns:
        Tuple of (args, formats, output_dir) — everything main() needs
        to boot the annotator.
    """
    parser = argparse.ArgumentParser(
        prog="circle_annotator",
        description=(
            "Circle Annotator -- interactive circle annotation for "
            "round scientific arenas.\n\n"
            "Click the centre of the arena, then click the rim to set the\n"
            "radius. Drag the centre to reposition and the rim handle to\n"
            "resize. Annotations are persisted as sidecar JSON files\n"
            "(resumable) and exported in COCO, YOLO, or Pascal VOC format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s -d ./images/ -a coco,yolo\n"
            "  %(prog)s -i photo.png -a yolo,coco\n"
            "  %(prog)s -f filelist.txt -o ./output/ -a voc\n"
            "\n"
            "key bindings (inside the annotation window):\n"
            "  1st left click       Set circle centre\n"
            "  2nd left click       Set radius (distance to centre)\n"
            "  Drag centre          Move circle\n"
            "  Drag rim handle      Resize circle\n"
            "  Right click          Delete annotation\n"
            "  Left / Right arrow   Previous / Next image\n"
            "  U / Tab              Jump to first unlabelled image\n"
            "  G                    Go to image number (type digits + Enter)\n"
            "  R                    Repeat circle from previous image\n"
            "  X                    Reset circle (with confirmation)\n"
            "  L                    Toggle labels\n"
            "  F                    Toggle circle fill\n"
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
    parser.add_argument(
        "-g", "--grid", action="store_true",
        help="Split view: annotator (left) + a 4x4 grid of pre-sampled context "
             "frames (right), ring shown live on all. Frames must be baked to "
             "<context-dir>/<image_stem>/*.png (no video handling here).",
    )
    parser.add_argument(
        "--context-dir", type=str, default=None,
        help="Root of the pre-sampled context frames (default: <image_dir>/context).",
    )

    args = parser.parse_args()

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

    return args, formats, output_dir


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ENTRY POINT                                  « start sequence »    │
# └─────────────────────────────────────────────────────────────────────┘
def main():
    """Parse CLI, discover images, print status, and launch the GUI."""
    args, formats, output_dir = parse_args()

    images = discover_images(
        directory=args.directory,
        image_path=args.image,
        filelist=args.filelist,
    )

    print("┌──────────────────────────────────────┐")
    print("│  CIRCLE ANNOTATOR v1.0               │")
    print("│  « click. drag. export. science. »   │")
    print("└──────────────────────────────────────┘")
    print(f"  Images:   {len(images)}")
    print(f"  Output:   {output_dir}")
    print(f"  Formats:  {', '.join(formats)}")
    print("  Press H inside the window for help.\n")

    CircleAnnotator(images, output_dir, formats, grid=args.grid, context_dir=args.context_dir)


if __name__ == "__main__":
    main()
