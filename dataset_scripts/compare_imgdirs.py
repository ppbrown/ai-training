#!/bin/env python
"""
Util to compare images in checkpoint save subdirs.
Give it two directories,and it will attempt to display sequential images
side by side.
This makes it easier to check if current run is better or worse than previous one
"""

import sys, os, argparse
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QMessageBox, QGridLayout
)
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtCore import Qt

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

def find_images_recursive(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(VALID_EXTENSIONS):
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, folder)
                image_files.append((rel_path, full_path))
    image_files.sort()
    return image_files

class ImageCompareViewer(QWidget):
    def __init__(self, folder1, folder2, folder3=None, skip=0):
        super().__init__()
        self.setWindowTitle("Recursive Image Comparator")
        self.resize(1040, 520)

        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3

        images1 = find_images_recursive(folder1)
        images2 = find_images_recursive(folder2)
        if folder3:
            self.folder3 = folder3
            images3 = find_images_recursive(folder3)
        else:
            images3 = None

        # Match by relative path
        common_keys = set(k for k, _ in images1) & set(k for k, _ in images2)
        if images3:
            common_keys &= set(k for k, _ in images3)
        common_keys = sorted(common_keys)

        self.pairs = [(dict(images1)[k], dict(images2)[k]) for k in common_keys]
        self.max_index = len(self.pairs)
        self.index = skip

        if self.index >= self.max_index:
           self.index = self.max_index - 1

        def makelabel(ndx: int):
            label1 = QLabel(f"Folder {ndx}")
            label1.setAlignment(Qt.AlignCenter)
            label1.setScaledContents(True)
            return label1

        self.label1 = makelabel(1)
        self.label2 = makelabel(2)
        if folder3:
            self.label3 = makelabel(3)

        layout = QGridLayout()
        layout.addWidget(self.label1, 0, 0)
        layout.addWidget(self.label2, 0, 1)
        if folder3:
            layout.addWidget(self.label3, 1, 0)
        self.setLayout(layout)

        if not self.pairs:
            QMessageBox.critical(self, "No matches", "No matching image filenames found.")
            sys.exit(1)

        self.update_images()

    def update_images(self):
        def prepare_pixmap(path):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                return QPixmap(512, 512)  # fallback blank
            pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # center-crop to 512×512
            x = max(0, (pixmap.width() - 512) // 2)
            y = max(0, (pixmap.height() - 512) // 2)
            return pixmap.copy(x, y, 512, 512)

        if 0 <= self.index < self.max_index:
            img1, img2 = self.pairs[self.index]
            self.label1.setPixmap(prepare_pixmap(img1))
            self.label2.setPixmap(prepare_pixmap(img2))
            if self.folder3:
                img3 = self.pairs[self.index]
                self.label3.setPixmap(prepare_pixmap(img3))

            #basename = os.path.basename(self.pairs[self.index][0])
            basename = self.pairs[self.index][0].removeprefix(self.folder1)
            self.setWindowTitle(f"[{self.index+1}/{self.max_index}] {basename}")
        else:
            QMessageBox.warning(self, "Out of range", "No more images.")


    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Right or key == Qt.Key_Space:
            self.index = min(self.index + 1, self.max_index - 1)
            self.update_images()
        elif key == Qt.Key_Left or key == Qt.Key_Backspace:
            self.index = max(self.index - 1, 0)
            self.update_images()
        elif key == Qt.Key_Escape or key == Qt.Key_Q:
            self.close()

def main():
    parser = argparse.ArgumentParser(description="Recursively compare image folders side-by-side.")
    parser.add_argument("folder1", help="First folder")
    parser.add_argument("folder2", help="Second folder")
    parser.add_argument("folder3", nargs="?", help="Optional third folder")
    parser.add_argument("--skip", type=int, default=0, help="Number of initial image matches to skip over")
    args = parser.parse_args()

    if not os.path.isdir(args.folder1) or not os.path.isdir(args.folder2):
        print("Error: One or both paths are not valid directories.")
        sys.exit(1)
    print("", args.folder1, "\n", args.folder2,
            f"\n{args.folder3}" if args.folder3 else "")

    app = QApplication(sys.argv)
    viewer = ImageCompareViewer(
            args.folder1, 
            args.folder2,
            args.folder3 if args.folder3 else None,
            args.skip)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
