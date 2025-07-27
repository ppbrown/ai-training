#!/bin/env python

# GUI, slightly fancier version of count_tags.py
# You just need  
#  PySide6>=6.6
# installed with pip


import sys
import os
from collections import Counter
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView
)

class TagCounterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Tag Counter")
        self.resize(600, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Directory chooser
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Directory:")
        self.dir_edit = QLineEdit()
        self.dir_btn = QPushButton("Browse")
        self.dir_btn.clicked.connect(self.choose_dir)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(self.dir_btn)
        layout.addLayout(dir_layout)

        # File extension and delimiter options
        opt_layout = QHBoxLayout()
        self.ext_label = QLabel("Tag file ext:")
        self.ext_edit = QLineEdit(".txt")
        self.delim_label = QLabel("Delimiter:")
        self.delim_edit = QLineEdit(",")
        opt_layout.addWidget(self.ext_label)
        opt_layout.addWidget(self.ext_edit)
        opt_layout.addWidget(self.delim_label)
        opt_layout.addWidget(self.delim_edit)
        layout.addLayout(opt_layout)

        # Run and Save buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Count Tags")
        self.run_btn.clicked.connect(self.run_count)
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        # Status label
        self.status = QLabel("")
        layout.addWidget(self.status)

        # Table to show results
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Tag", "Count"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.tag_counts = {}

    def choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            self.dir_edit.setText(path)

    def run_count(self):
        directory = self.dir_edit.text().strip()
        ext = self.ext_edit.text().strip() or ".txt"
        delim = self.delim_edit.text().strip() or ","
        if not directory or not os.path.isdir(directory):
            self.status.setText("Please select a valid directory.")
            return
        
        self.status.setText("Counting tags...")
        QApplication.processEvents()

        tag_counter = Counter()

        files = []
        for root, dirs, filenames in os.walk(directory):
            for fname in filenames:
                if fname.endswith(ext):
                    files.append(os.path.join(root, fname))
        if not files:
            self.status.setText(f"No '{ext}' files found.")
            return

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    line = f.read().strip()
                    tags = [t.strip() for t in line.split(delim) if t.strip()]
                    tag_counter.update(tags)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error reading {filepath}: {e}")

        self.tag_counts = dict(tag_counter.most_common())
        self.update_table()
        self.status.setText(f"Done. {len(self.tag_counts)} unique tags found.")
        self.save_btn.setEnabled(True)

    def update_table(self):
        self.table.setRowCount(0)
        for tag, count in self.tag_counts.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(tag))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))

    def save_results(self):
        if not self.tag_counts:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Tag Counts", "tag_counts.txt", "Text Files (*.txt)")
        if fname:
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    for tag, count in self.tag_counts.items():
                        f.write(f"{tag}\t{count}\n")
                self.status.setText(f"Results saved to {fname}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error saving file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TagCounterGUI()
    gui.show()
    sys.exit(app.exec())
