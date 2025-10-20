"""Filter panel UI."""

from __future__ import annotations
from typing import Callable, Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QComboBox,
)


class FilterPanel(QWidget):
    """Sidebar with live controls for filters."""

    def __init__(
        self,
        on_toggle_baf: Callable[[bool], None],
        on_change_baf: Callable[[int, int, int, int], None],
        on_toggle_refractory: Callable[[bool], None],
        on_change_refractory: Callable[[int], None],
        on_select_visual: Callable[[str | None], None],
        on_change_visual_params: Callable[[dict], None],
        visual_filters: Iterable[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        # BAF group
        gb_baf = QGroupBox("Background Activity Filter")
        vb = QVBoxLayout(gb_baf)
        self.chk_baf = QCheckBox("Enable BAF")
        self.chk_baf.stateChanged.connect(lambda s: on_toggle_baf(s == Qt.Checked))
        vb.addWidget(self.chk_baf)

        # window_ms
        hb1 = QHBoxLayout()
        hb1.addWidget(QLabel("window_ms"))
        self.sp_baf_win = QSpinBox()
        self.sp_baf_win.setRange(1, 2000)
        self.sp_baf_win.setValue(50)
        hb1.addWidget(self.sp_baf_win)
        vb.addLayout(hb1)

        # count_threshold
        hb2 = QHBoxLayout()
        hb2.addWidget(QLabel("count_threshold"))
        self.sp_baf_cnt = QSpinBox()
        self.sp_baf_cnt.setRange(1, 20)
        self.sp_baf_cnt.setValue(1)
        hb2.addWidget(self.sp_baf_cnt)
        vb.addLayout(hb2)

        # refractory_us
        hb3 = QHBoxLayout()
        hb3.addWidget(QLabel("refractory_us"))
        self.sp_baf_ref = QSpinBox()
        self.sp_baf_ref.setRange(0, 200000)
        self.sp_baf_ref.setValue(500)
        hb3.addWidget(self.sp_baf_ref)
        vb.addLayout(hb3)

        # spatial_radius
        hb4 = QHBoxLayout()
        hb4.addWidget(QLabel("spatial_radius"))
        self.sp_baf_rad = QSpinBox()
        self.sp_baf_rad.setRange(0, 5)
        self.sp_baf_rad.setValue(1)
        hb4.addWidget(self.sp_baf_rad)
        vb.addLayout(hb4)

        # Connect all BAF param changes
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

        # Refractory group
        gb_ref = QGroupBox("Refractory (per pixel)")
        vb2 = QVBoxLayout(gb_ref)
        self.chk_ref = QCheckBox("Enable Refractory")
        self.chk_ref.stateChanged.connect(lambda s: on_toggle_refractory(s == Qt.Checked))
        vb2.addWidget(self.chk_ref)

        hb5 = QHBoxLayout()
        hb5.addWidget(QLabel("refractory_us"))
        self.sp_ref_us = QSpinBox()
        self.sp_ref_us.setRange(0, 200000)
        self.sp_ref_us.setValue(500)
        self.sp_ref_us.valueChanged.connect(lambda _=None: on_change_refractory(self.sp_ref_us.value()))
        hb5.addWidget(self.sp_ref_us)
        vb2.addLayout(hb5)

        root.addWidget(gb_ref)

        # Visualisation group
        gb_vis = QGroupBox("Visualisierung")
        vb3 = QVBoxLayout(gb_vis)
        hb6 = QHBoxLayout()
        hb6.addWidget(QLabel("Filter"))
        self.cmb_visual = QComboBox()
        self.cmb_visual.addItem("Ohne Filter", userData=None)
        self._visual_filters = list(visual_filters or [])
        for name in self._visual_filters:
            self.cmb_visual.addItem(name, userData=name)
        hb6.addWidget(self.cmb_visual)
        vb3.addLayout(hb6)

        hb7 = QHBoxLayout()
        self.lbl_event_count = QLabel("Integration (ms)")
        self.sp_event_count = QSpinBox()
        self.sp_event_count.setRange(1, 5000)
        self.sp_event_count.setValue(50)
        hb7.addWidget(self.lbl_event_count)
        hb7.addWidget(self.sp_event_count)
        vb3.addLayout(hb7)

        self.lbl_event_count.setVisible(False)
        self.sp_event_count.setVisible(False)

        def _on_visual_changed(index: int) -> None:
            name = self.cmb_visual.itemData(index)
            on_select_visual(name)
            show_event_count = name == "Event Count Image"
            self.lbl_event_count.setVisible(show_event_count)
            self.sp_event_count.setVisible(show_event_count)
            if show_event_count:
                on_change_visual_params({"integration_ms": self.sp_event_count.value()})

        self.cmb_visual.currentIndexChanged.connect(_on_visual_changed)
        self.sp_event_count.valueChanged.connect(
            lambda val: on_change_visual_params({"integration_ms": val})
        )

        root.addWidget(gb_vis)
        root.addStretch(1)
