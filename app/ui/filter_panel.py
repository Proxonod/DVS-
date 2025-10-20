"""Filter panel UI.

This module defines ``FilterPanel``, a simple sidebar widget that
allows users to enable or disable filters and adjust their
parameters in real time.  Each supported filter is presented as a
group box with a checkbox for toggling and spin boxes for tuning
parameters.  The panel communicates changes back to the main
window via callback functions passed at construction time.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QSpinBox,
    QGroupBox,
)


class FilterPanel(QWidget):
    """Sidebar with live controls for filters."""

    def __init__(
        self,
        on_toggle_baf: Callable[[bool], None],
        on_change_baf: Callable[[int, int, int, int], None],
        on_toggle_refractory: Callable[[bool], None],
        on_change_refractory: Callable[[int], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        # BAF controls
        gb_baf = QGroupBox("Background Activity Filter")
        vbox_baf = QVBoxLayout(gb_baf)
        self.chk_baf = QCheckBox("Enable BAF")
        self.chk_baf.stateChanged.connect(lambda s: on_toggle_baf(s == Qt.Checked))
        vbox_baf.addWidget(self.chk_baf)

        # window_ms
        hb1 = QHBoxLayout()
        hb1.addWidget(QLabel("window_ms"))
        self.sp_baf_win = QSpinBox()
        self.sp_baf_win.setRange(1, 2000)
        self.sp_baf_win.setValue(50)
        hb1.addWidget(self.sp_baf_win)
        vbox_baf.addLayout(hb1)

        # count_threshold
        hb2 = QHBoxLayout()
        hb2.addWidget(QLabel("count_threshold"))
        self.sp_baf_cnt = QSpinBox()
        self.sp_baf_cnt.setRange(1, 20)
        self.sp_baf_cnt.setValue(1)
        hb2.addWidget(self.sp_baf_cnt)
        vbox_baf.addLayout(hb2)

        # refractory_us
        hb3 = QHBoxLayout()
        hb3.addWidget(QLabel("refractory_us"))
        self.sp_baf_ref = QSpinBox()
        self.sp_baf_ref.setRange(0, 200000)
        self.sp_baf_ref.setValue(500)
        hb3.addWidget(self.sp_baf_ref)
        vbox_baf.addLayout(hb3)

        # spatial_radius
        hb4 = QHBoxLayout()
        hb4.addWidget(QLabel("spatial_radius"))
        self.sp_baf_rad = QSpinBox()
        self.sp_baf_rad.setRange(0, 5)
        self.sp_baf_rad.setValue(1)
        hb4.addWidget(self.sp_baf_rad)
        vbox_baf.addLayout(hb4)

        # Connect all BAF parameter changes
        for sp in (self.sp_baf_win, self.sp_baf_cnt, self.sp_baf_ref, self.sp_baf_rad):
            sp.valueChanged.connect(
                lambda _=None: on_change_baf(
                    self.sp_baf_win.value(),
                    self.sp_baf_cnt.value(),
                    self.sp_baf_ref.value(),
                    self.sp_baf_rad.value(),
                )
            )

        root.addWidget(gb_baf)

        # Refractory controls
        gb_ref = QGroupBox("Refractory (per pixel)")
        vbox_ref = QVBoxLayout(gb_ref)
        self.chk_ref = QCheckBox("Enable Refractory")
        self.chk_ref.stateChanged.connect(lambda s: on_toggle_refractory(s == Qt.Checked))
        vbox_ref.addWidget(self.chk_ref)

        hb5 = QHBoxLayout()
        hb5.addWidget(QLabel("refractory_us"))
        self.sp_ref_us = QSpinBox()
        self.sp_ref_us.setRange(0, 200000)
        self.sp_ref_us.setValue(500)
        self.sp_ref_us.valueChanged.connect(
            lambda _=None: on_change_refractory(self.sp_ref_us.value())
        )
        hb5.addWidget(self.sp_ref_us)
        vbox_ref.addLayout(hb5)

        root.addWidget(gb_ref)
        root.addStretch(1)
