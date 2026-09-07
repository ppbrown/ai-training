#!/usr/bin/env python3

"""
View a list of images one at a time, next to their matching caption file.
Image list comes from positional args, or from stdin if none are given.
"""

import sys
import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="View images one at a time next to their caption text."
    )
    parser.add_argument(
        "images", nargs="*",
        help="Image file paths (if omitted, read newline-separated paths from stdin)"
    )
    parser.add_argument(
        "--suffix", default="txt",
        help="Caption file extension to match (default: txt)"
    )
    args = parser.parse_args()
    suffix = args.suffix
    if not suffix.startswith("."):
        suffix = "." + suffix
    args.suffix = suffix
    return args


def get_image_list(args):
    if args.images:
        return [Path(p) for p in args.images]
    paths = []
    for line in sys.stdin:
        path = line.strip()
        if path:
            paths.append(Path(path))
    return paths


def caption_path_for(image_path, suffix):
    return image_path.with_suffix(suffix)


def build_viewer_class():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QLabel, QTextEdit, QGroupBox,
        QHBoxLayout, QVBoxLayout, QPushButton,
    )

    class Viewer(QMainWindow):
        def __init__(self, images, suffix, show_count=False):
            super().__init__()
            self.images = list(images)
            self.suffix = suffix
            self.show_count = show_count
            self.idx = 0

            self.setWindowTitle("Caption Viewer")

            self.filename_label = QLabel()
            self.filename_label.setAlignment(Qt.AlignLeft)

            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self.image_label.setStyleSheet("QLabel { border: 2px solid black; }")

            self.caption_edit = QTextEdit()
            self.caption_edit.setReadOnly(True)
            self.caption_edit.setAcceptRichText(False)
            self.caption_edit.setFixedWidth(512)
            self.caption_edit.setStyleSheet(
                "QTextEdit { background: #111; color: #ddd; padding: 6px; "
                "font-family: Consolas, 'Courier New', monospace; font-size: 12px; }"
            )

            self.prev_btn = QPushButton("Prev")
            self.next_btn = QPushButton("Next")
            self.prev_btn.clicked.connect(self.prev_item)
            self.next_btn.clicked.connect(self.next_item)

            nav_group = QGroupBox("Navigate")
            nav_layout = QHBoxLayout(nav_group)
            nav_layout.addWidget(self.prev_btn)
            nav_layout.addWidget(self.next_btn)

            self.delete_btn = QPushButton("Delete")
            self.delete_btn.clicked.connect(self.delete_current)

            self.quit_btn = QPushButton("Quit")
            self.quit_btn.clicked.connect(self.close)

            controls = QWidget()
            controls_layout = QHBoxLayout(controls)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.addWidget(nav_group)
            controls_layout.addStretch(1)
            controls_layout.addWidget(self.delete_btn)
            controls_layout.addStretch(1)
            controls_layout.addWidget(self.quit_btn)

            central = QWidget()
            main_layout = QVBoxLayout(central)
            main_layout.addWidget(self.filename_label)
            row = QHBoxLayout()
            row.addWidget(self.image_label)
            row.addSpacing(8)
            row.addWidget(self.caption_edit)
            main_layout.addLayout(row)
            main_layout.addSpacing(8)
            main_layout.addWidget(controls)

            self.setCentralWidget(central)

            if not self.images:
                self._set_empty()
            else:
                self._load_index(0)

            QShortcut(Qt.Key_Left, self, activated=self.prev_item)
            QShortcut(Qt.Key_Right, self, activated=self.next_item)
            QShortcut(Qt.Key_Q, self, activated=self.close)

        def _set_empty(self):
            self.filename_label.setText("")
            self.image_label.setFixedSize(512, 512)
            self.image_label.clear()
            self.image_label.setText("No images.")
            self.caption_edit.setPlainText("")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)

        def _load_index(self, i):
            self.idx = max(0, min(i, len(self.images) - 1))
            image_path = self.images[self.idx]

            pix = QPixmap(str(image_path))
            if pix.isNull():
                self.image_label.setFixedSize(512, 512)
                self.image_label.clear()
                self.image_label.setText(f"Failed to load image:\n{image_path}")
            else:
                pix_scaled = pix.scaled(
                    512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setFixedSize(pix_scaled.size())
                self.image_label.setPixmap(pix_scaled)

            if self.show_count:
                self.filename_label.setText(
                    f"{image_path.name}  ({self.idx + 1}/{len(self.images)})"
                )
            else:
                self.filename_label.setText(image_path.name)
            caption_path = caption_path_for(image_path, self.suffix)
            self.caption_edit.setPlainText(self._read_text(caption_path))

            self.prev_btn.setEnabled(self.idx > 0)
            self.next_btn.setEnabled(self.idx < len(self.images) - 1)
            self.delete_btn.setEnabled(True)

        def _read_text(self, path):
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return f"[Error reading {path.name}: {e}]"

        def next_item(self):
            if self.idx < len(self.images) - 1:
                self._load_index(self.idx + 1)

        def prev_item(self):
            if self.idx > 0:
                self._load_index(self.idx - 1)

        def delete_current(self):
            if not self.images:
                return
            path = self.images[self.idx]
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error deleting {path}: {e}", file=sys.stderr)
                return
            del self.images[self.idx]
            if not self.images:
                self._set_empty()
            else:
                self._load_index(min(self.idx, len(self.images) - 1))

    return Viewer


def main():
    args = parse_args()
    show_count = bool(args.images)
    images = get_image_list(args)

    Viewer = build_viewer_class()
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = Viewer(images, args.suffix, show_count)
    win.resize(1050, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
