#!/usr/bin/env python3

"""
I made this utility, so that when I create competing caption files
(eg: moondream in .moon, and qwen in .txt)
I can look at each image, with its matching captions, side-by-side
"""

import sys
import os
import argparse
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextOption, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QTextEdit,
    QHBoxLayout, QVBoxLayout, QPushButton, QStatusBar
)


def find_triplets(args):
    """
    Yield (jpg_path, txt_path, moon_path) for files sharing the same stem.
    Only '.jpg' is considered for the image extension as requested.
    """
    root = args.folder
    root = root.resolve()
    # Map of stem -> list of suffixes present
    stems = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            suf = p.suffix.lower()
            if suf not in (".jpg", args.suffix1, args.suffix2):
                continue
            stem = p.with_suffix("").name  # local stem without extension
            stems.setdefault((Path(dirpath), stem), set()).add(suf)

    for (d, stem), sufs in stems.items():
        if {".jpg", args.suffix1, args.suffix2}.issubset(sufs):
            yield (
                d / f"{stem}.jpg",
                d / f"{stem}{args.suffix1}",
                d / f"{stem}{args.suffix2}",
            )


class Viewer(QMainWindow):
    def __init__(self, triplets):
        super().__init__()
        self.triplets = list(triplets)
        self.idx = 0

        self.setWindowTitle("JPG + TXT + MOON Viewer")

        # --- Left: image ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.image_label.setStyleSheet("QLabel { background: #222; }")

        # --- Right: fixed 512x512 panel with two 512x256 text widgets ---
        self.right_panel = QWidget()
        self.right_panel.setFixedSize(512, 512)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.txt_edit = self._make_text_box(height=256)
        self.moon_edit = self._make_text_box(height=256)

        right_layout.addWidget(self.txt_edit)
        right_layout.addWidget(self.moon_edit)

        # --- Controls ---
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.quit_btn = QPushButton("Quit")
        self.prev_btn.clicked.connect(self.prev_item)
        self.next_btn.clicked.connect(self.next_item)
        self.quit_btn.clicked.connect(self.quit)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.prev_btn)
        controls_layout.addWidget(self.next_btn)
        controls_layout.addWidget(self.quit_btn)
        controls_layout.addStretch(1)

        # --- Main layout ---
        central = QWidget()
        main_layout = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(self.image_label)
        row.addSpacing(8)
        row.addWidget(self.right_panel)
        main_layout.addLayout(row)
        main_layout.addSpacing(8)
        main_layout.addWidget(controls)

        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        if not self.triplets:
            self._set_empty()
        else:
            self._load_index(0)

        # Global key shortcuts (active whenever the window is active)
        QShortcut(Qt.Key_Right, self, activated=self.next_item)
        QShortcut(Qt.Key_Down, self, activated=self.next_item)
        QShortcut(Qt.Key_Left, self, activated=self.prev_item)
        QShortcut(Qt.Key_Up, self, activated=self.prev_item)
        QShortcut(Qt.Key_Q, self, activated=self.quit)

    def _make_text_box(self, height: int) -> QTextEdit:
        te = QTextEdit()
        te.setReadOnly(True)
        te.setAcceptRichText(False)
        te.setFixedSize(512, height)
        te.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        te.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        te.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        te.setFrameStyle(QtWidgets.QFrame.NoFrame)
        te.setStyleSheet(
            "QTextEdit { background: #111; color: #ddd; padding: 6px; font-family: Consolas, 'Courier New', monospace; font-size: 12px; }"
        )
        return te

    def _set_empty(self):
        self.image_label.setText("No matching (jpg, txt, moon) triplets found.")
        self.txt_edit.setPlainText("")
        self.moon_edit.setPlainText("")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.status.showMessage("0 / 0")

    def _load_index(self, i: int):
        self.idx = max(0, min(i, len(self.triplets) - 1))
        jpg_path, txt_path, moon_path = self.triplets[self.idx]

        # Load and scale image to <= 512 width, keep aspect ratio
        pix = QPixmap(str(jpg_path))
        if pix.isNull():
            # Fallback: clear if failing to load
            self.image_label.clear()
            self.image_label.setText(f"Failed to load image:\n{jpg_path}")
        else:
            target_w = min(512, pix.width())
            pix_scaled = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
            self.image_label.setPixmap(pix_scaled)

        # Load texts; overflow will be clipped (no scrollbars)
        self.txt_edit.setPlainText(self._read_text(txt_path))
        self.moon_edit.setPlainText(self._read_text(moon_path))

        # Update buttons + status
        self.prev_btn.setEnabled(self.idx > 0)
        self.next_btn.setEnabled(self.idx < len(self.triplets) - 1)
        self.status.showMessage(f"{self.idx + 1} / {len(self.triplets)}  —  {jpg_path}")

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[Error reading {path.name}: {e}]"

    def next_item(self):
        if self.idx < len(self.triplets) - 1:
            self._load_index(self.idx + 1)

    def prev_item(self):
        if self.idx > 0:
            self._load_index(self.idx - 1)

    def quit(self):
        exit(0)


def main():
    parser = argparse.ArgumentParser(description="Recursively compare images with matching caption files.")
    parser.add_argument("--suffix1", default=".txt", help="First captiontype(default=.txt)")
    parser.add_argument("--suffix2", default=".moon", help="Second caption type(default=.moon)")
    parser.add_argument("folder", help="Directory with images and matching caption files")
    args = parser.parse_args()

    triplets = list(find_triplets(args))

    app = QApplication(sys.argv)
    win = Viewer(triplets)
    win.resize(512 + 8 + 512, 700)  # a reasonable starting size
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
