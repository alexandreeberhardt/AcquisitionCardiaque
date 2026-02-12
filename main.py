import sys
import csv
import logging
from collections import deque

import nidaqmx
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from scipy.signal import butter, filtfilt

import ecg_detectors_modified


logger = logging.getLogger(__name__)


class VisualiseurSignal(QtWidgets.QMainWindow):
    """
    Visualiseur / Acquisiteur ECG avec plusieurs fonctions :

    - Acquisition NI-DAQ (USB-6000) OU lecture fichier
    - Affichage en mode scrolling ou lecture simple centrée sur le dernier pic
    - Mesure Δt entre 2 curseurs + stats amplitude + curseur horizontal
    - Filtres optionnels (passe-bas / passe-haut / moyenne glissante)
    - Détection pics (algo de Pan-Tompkins) + BPM
    - Capture d'écran + enregistrement CSV (uniquement en live)

    - Les électrodes inversées et les filtres peuvent être déployer à l'affichage
    - La détection des pics est faite sur un timer séparé, plus lent, pour garder une UI fluide même quand le buffer grandit
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Acquisition du signal en temps réel")

        # -----------------------
        #Paramètres principaux

        self.fs = 1000  #Hz
        self.taille_lot = 50  #échantillons par cycle acquisition
        self.plot_refresh_ms = 30  #ms (rafraîchissement affichage)
        self.detect_refresh_ms = 150  #ms (détection pics)
        self.max_points = 5000  #points max conservés en mémoire affichage
        self.detect_window_s = 10.0  #fenêtre de détection
        self.window_scroll_s = 10.0  #fenêtre affichage (scroll)
        self.simple_window_s = 1.0  #fenêtre affichage (simple)

        #Lecture simple = maintien du centrage après un pic
        self.simple_hold_ms = 200
        self.simple_hold_steps = 1  #recalculé avec le timer acquisition
        self.simple_hold_counter = 0
        self.simple_hold_center = None
        self.simple_last_peak_time = None

        # ---------------------------------------------------------------------
        #État et sources

        self.channels = []
        self.current_channel = None
        self.mode_fichier = False
        self.en_pause = False

        self.task = None  #nidaqmx.Task (mode live)
        self.pointer = 0  #lecture fichier
        self.current_time = 0.0  #live

        #Données fichier
        self.donnees_x = []
        self.donnees_y = []

        #Buffer temps + amplitude
        #On applique inversion et filtres à l'affichage, pas dans l'acquisition
        self.full_x = deque(maxlen=self.max_points)
        self.full_y = deque(maxlen=self.max_points)

        #Enregistrement (mode live uniquement)
        self.is_recording = False
        self.recording_file = None
        self.recording_writer = None

        #Mode d'affichage
        self.current_display_mode = "scroll"  # scroll - simple

        #Cache pics calculés par des timer séparé
        self._pics_x = np.array([], dtype=float)
        self._pics_y = np.array([], dtype=float)

        #UI
        self.initialiser_interface()

        #Timers
        self.timer_acq = QtCore.QTimer(self)
        self.timer_acq.timeout.connect(self.acquisition_cycle)

        self.timer_plot = QtCore.QTimer(self)
        self.timer_plot.timeout.connect(self.affichage_cycle)

        self.timer_detect = QtCore.QTimer(self)
        self.timer_detect.timeout.connect(self.detect_peaks)

        self._recalculer_hold_steps()

        #Init canaux
        self.reset_interface()

    # ---------------------------------------------------------------------
    #UI

    def initialiser_interface(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        #Sélection du canal
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItems(self.channels)
        self.chan_combo.currentIndexChanged.connect(self.change_channel)

        ligne_canal = QtWidgets.QHBoxLayout()
        ligne_canal.addWidget(QtWidgets.QLabel("Canal AI :"))
        ligne_canal.addWidget(self.chan_combo, 1)

        self.btn_inverted = QtWidgets.QCheckBox("Électrodes inversées")
        ligne_canal.addWidget(self.btn_inverted)

        layout.addLayout(ligne_canal)

        #Plot
        self.plot = pg.PlotWidget(title="Signal en temps réel")
        self.plot.setLabel("left", "Amplitude (V)")
        self.plot.setLabel("bottom", "Temps (s)")
        self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        layout.addWidget(self.plot, 1)

        self.curve = self.plot.plot(pen=pg.mkPen("y", width=1))
        self.peaks = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("r"))
        self.plot.addItem(self.peaks)

        self.l1 = pg.InfiniteLine(0.1, angle=90, movable=True, pen=pg.mkPen("r", width=1))
        self.l2 = pg.InfiniteLine(0.2, angle=90, movable=True, pen=pg.mkPen("r", width=1))
        self.plot.addItem(self.l1)
        self.plot.addItem(self.l2)
        self.l1.sigPositionChanged.connect(self.actualiser_mesure)
        self.l2.sigPositionChanged.connect(self.actualiser_mesure)

        self.lamp = pg.InfiniteLine(0.0, angle=0, movable=True, pen=pg.mkPen("c", width=1))
        self.plot.addItem(self.lamp)
        self.lamp.sigPositionChanged.connect(self.actualiser_stats)

        #Infos
        ligne_infos = QtWidgets.QHBoxLayout()
        self.lbl_dt = QtWidgets.QLabel("Delta temps: 0.000 s")
        self.lbl_stat = QtWidgets.QLabel("Amplitude stats: N/A")
        self.lbl_bpm = QtWidgets.QLabel("BPM : --")
        self.lbl_peak = QtWidgets.QLabel("Dernier pic : -- V")

        ligne_infos.addWidget(self.lbl_dt)
        ligne_infos.addWidget(self.lbl_stat, 1)
        ligne_infos.addWidget(self.lbl_bpm)
        ligne_infos.addWidget(self.lbl_peak)
        layout.addLayout(ligne_infos)

        #Contrôles
        h_ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(h_ctrl)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        h_ctrl.addWidget(self.btn_pause)

        self.btn_stop = QtWidgets.QPushButton("Arrêter")
        self.btn_stop.clicked.connect(self.stop_all)
        h_ctrl.addWidget(self.btn_stop)

        self.btn_reset = QtWidgets.QPushButton("Redémarrer")
        self.btn_reset.clicked.connect(self.reset_interface)
        h_ctrl.addWidget(self.btn_reset)

        self.btn_capture = QtWidgets.QPushButton("Capture d'écran")
        self.btn_capture.clicked.connect(self.capture_ecran)
        h_ctrl.addWidget(self.btn_capture)

        self.btn_record = QtWidgets.QPushButton("Enregistrer")
        self.btn_record.setCheckable(True)
        self.btn_record.setChecked(False)
        self.btn_record.toggled.connect(self.toggle_recording)
        h_ctrl.addWidget(self.btn_record)

        #Mode d'affichage
        self.grp_modes = QtWidgets.QGroupBox("Mode d'affichage")
        mode_layout = QtWidgets.QHBoxLayout(self.grp_modes)
        self.rbtn_scroll = QtWidgets.QRadioButton("Défilement (scrolling)")
        self.rbtn_simple = QtWidgets.QRadioButton("Lecture simple")
        self.rbtn_scroll.setChecked(True)

        mode_layout.addWidget(self.rbtn_scroll)
        mode_layout.addWidget(self.rbtn_simple)
        layout.addWidget(self.grp_modes)

        self.rbtn_scroll.toggled.connect(self._on_mode_change)
        self.rbtn_simple.toggled.connect(self._on_mode_change)

        #Filtres
        self.grp_filtres = QtWidgets.QGroupBox("Filtres (option)")
        filtre_layout = QtWidgets.QVBoxLayout(self.grp_filtres)

        ligne_chk = QtWidgets.QHBoxLayout()
        self.chk_filtre_passe_bas = QtWidgets.QCheckBox("Passe-bas")
        self.chk_filtre_passe_haut = QtWidgets.QCheckBox("Passe-haut")
        self.chk_filtre_moyenne = QtWidgets.QCheckBox("Moyenne glissante")
        ligne_chk.addWidget(self.chk_filtre_passe_bas)
        ligne_chk.addWidget(self.chk_filtre_passe_haut)
        ligne_chk.addWidget(self.chk_filtre_moyenne)
        ligne_chk.addStretch(1)
        filtre_layout.addLayout(ligne_chk)

        ligne_params = QtWidgets.QGridLayout()
        filtre_layout.addLayout(ligne_params)

        self.lbl_cutoff1 = QtWidgets.QLabel("Fréquence de coupure PB (Hz) :")
        self.spin_cutoff1 = QtWidgets.QDoubleSpinBox()
        self.spin_cutoff1.setRange(0.1, 500.0)
        self.spin_cutoff1.setValue(10.0)
        self.spin_cutoff1.setSingleStep(0.5)

        self.lbl_cutoff2 = QtWidgets.QLabel("Fréquence de coupure PH (Hz) :")
        self.spin_cutoff2 = QtWidgets.QDoubleSpinBox()
        self.spin_cutoff2.setRange(0.1, 500.0)
        self.spin_cutoff2.setValue(0.5)
        self.spin_cutoff2.setSingleStep(0.1)

        self.lbl_order = QtWidgets.QLabel("Ordre :")
        self.spin_order = QtWidgets.QSpinBox()
        self.spin_order.setRange(1, 10)
        self.spin_order.setValue(4)

        self.spin_fenetre_lbl = QtWidgets.QLabel("Fenêtre moyenne glissante :")
        self.spin_fenetre_moyenne = QtWidgets.QSpinBox()
        self.spin_fenetre_moyenne.setRange(1, 200)
        self.spin_fenetre_moyenne.setValue(5)

        ligne_params.addWidget(self.spin_fenetre_lbl, 0, 0)
        ligne_params.addWidget(self.spin_fenetre_moyenne, 0, 1)
        ligne_params.addWidget(self.lbl_cutoff1, 0, 2)
        ligne_params.addWidget(self.spin_cutoff1, 0, 3)
        ligne_params.addWidget(self.lbl_cutoff2, 0, 4)
        ligne_params.addWidget(self.spin_cutoff2, 0, 5)
        ligne_params.addWidget(self.lbl_order, 0, 6)
        ligne_params.addWidget(self.spin_order, 0, 7)
        ligne_params.setColumnStretch(8, 1)

        layout.addWidget(self.grp_filtres)

        #Source (live ou alors fichier)
        ligne_source = QtWidgets.QHBoxLayout()
        self.btn_live = QtWidgets.QPushButton("Acquisition en direct")
        self.btn_live.clicked.connect(self.start_live)
        ligne_source.addWidget(self.btn_live)

        self.btn_file = QtWidgets.QPushButton("Lecture fichier")
        self.btn_file.clicked.connect(self.start_file)
        ligne_source.addWidget(self.btn_file)

        ligne_source.addStretch(1)
        layout.addLayout(ligne_source)

        #Vue d'ensemble pour le fichier
        self.plot_overview = pg.PlotWidget(title="Vue d'ensemble")
        self.plot_overview.setMaximumHeight(260)
        self.curve_overview = self.plot_overview.plot(pen=pg.mkPen("w", width=1))
        self.cursor_overview = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y", width=1))
        self.plot_overview.addItem(self.cursor_overview)
        self.plot_overview.hide()
        layout.addWidget(self.plot_overview)

        #on force un redraw dès qu'on sélectionne des filtres ou une inversion
        self.btn_inverted.stateChanged.connect(self._forcer_refresh)
        self.chk_filtre_passe_bas.stateChanged.connect(self._forcer_refresh)
        self.chk_filtre_passe_haut.stateChanged.connect(self._forcer_refresh)
        self.chk_filtre_moyenne.stateChanged.connect(self._forcer_refresh)

        self.spin_cutoff1.valueChanged.connect(self._forcer_refresh)
        self.spin_cutoff2.valueChanged.connect(self._forcer_refresh)
        self.spin_order.valueChanged.connect(self._forcer_refresh)
        self.spin_fenetre_moyenne.valueChanged.connect(self._forcer_refresh)

    # ---------------------------------------------------------------------
    #Modes et changements UI

    def _on_mode_change(self, checked: bool):
        if not checked:
            return
        self.set_display_mode("scroll" if self.rbtn_scroll.isChecked() else "simple")
        self._forcer_refresh()

    def _forcer_refresh(self):
        self.affichage_cycle()

    def set_display_mode(self, mode):
        if mode == "scroll":
            self.current_display_mode = "scroll"
            self.rbtn_scroll.setChecked(True)
            self.rbtn_simple.setChecked(False)
        else:
            self.current_display_mode = "simple"
            self.rbtn_simple.setChecked(True)
            self.rbtn_scroll.setChecked(False)

    # ---------------------------------------------------------------------
    #NI

    def change_channel(self, idx):
        if not self.channels or idx < 0 or idx >= len(self.channels):
            return

        self.current_channel = self.channels[idx]

        if self.mode_fichier:
            return

        etait_en_cours = self.task is not None
        try:
            if self.task:
                self.task.stop()
                self.task.close()
        except Exception:
            pass

        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(self.current_channel)
        self.task.timing.cfg_samp_clk_timing(
            rate=self.fs,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
        )

        #Reset buffers
        self.full_x.clear()
        self.full_y.clear()
        self.current_time = 0.0
        self.simple_last_peak_time = None
        self.simple_hold_counter = 0
        self.simple_hold_center = None
        self._pics_x = np.array([], dtype=float)
        self._pics_y = np.array([], dtype=float)

        if etait_en_cours:
            try:
                self.task.start()
            except Exception as e:
                logger.exception("Impossible de redémarrer la tâche NI: %s", e)

    # ---------------------------------------------------------------------
    #Contrôles

    def toggle_pause(self):
        self.en_pause = not self.en_pause
        self.btn_pause.setText("Lancer" if self.en_pause else "Pause")

        if self.en_pause:
            self.timer_acq.stop()
            self.timer_plot.stop()
            self.timer_detect.stop()
        else:
            self.timer_acq.start(self._acq_interval_ms())
            self.timer_plot.start(self.plot_refresh_ms)
            self.timer_detect.start(self.detect_refresh_ms)

    def stop_all(self):
        self.timer_acq.stop()
        self.timer_plot.stop()
        self.timer_detect.stop()

        try:
            if self.task:
                self.task.stop()
                self.task.close()
        except Exception:
            pass
        self.task = None

        self.is_recording = False
        self.stop_recording()

        self.full_x.clear()
        self.full_y.clear()

        self.donnees_x = []
        self.donnees_y = []
        self.pointer = 0
        self.current_time = 0.0

        self.simple_hold_counter = 0
        self.simple_last_peak_time = None
        self.simple_hold_center = None

        self._pics_x = np.array([], dtype=float)
        self._pics_y = np.array([], dtype=float)

        self.curve.clear()
        self.peaks.clear()

        self.lbl_bpm.setText("BPM : --")
        self.lbl_peak.setText("Dernier pic : -- V")
        self.lbl_dt.setText("Delta temps: 0.000 s")
        self.lbl_stat.setText("Amplitude stats: N/A")

        self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.plot_overview.hide()

        self.btn_live.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.chan_combo.setEnabled(True)
        self.btn_record.setEnabled(True)

        self.set_display_mode("scroll")
        self.mode_fichier = False
        self.en_pause = False
        self.btn_pause.setText("Pause")

    def reset_interface(self):
        self.stop_all()
        try:
            devs = nidaqmx.system.System.local().devices
            usb_devices = [d for d in devs if d.product_type == "USB-6000"]
            if usb_devices:
                device = usb_devices[0]
                self.channels = [c.name for c in device.ai_physical_chans]
                self.chan_combo.clear()
                self.chan_combo.addItems(self.channels)
                self.chan_combo.setCurrentIndex(0)
                self.current_channel = self.channels[0]
        except Exception:
            self.channels = []
            self.chan_combo.clear()

    def start_live(self):
        self.stop_all()
        self.mode_fichier = False

        try:
            devs = nidaqmx.system.System.local().devices
            usb_devices = [d for d in devs if d.product_type == "USB-6000"]
            if not usb_devices:
                raise Exception("Aucun boîtier NI USB-6000 détecté. Branchez-le et réessayez.")
            device = usb_devices[0]

            self.channels = [c.name for c in device.ai_physical_chans]
            self.chan_combo.clear()
            self.chan_combo.addItems(self.channels)
            self.chan_combo.setCurrentIndex(0)
            self.current_channel = self.channels[0]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur NI", str(e))
            return

        self.change_channel(self.chan_combo.currentIndex())

        try:
            self.task.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur NI", f"Impossible de démarrer l'acquisition :\n{e}")
            self.stop_all()
            return

        self.timer_acq.start(self._acq_interval_ms())
        self.timer_plot.start(self.plot_refresh_ms)
        self.timer_detect.start(self.detect_refresh_ms)

    def start_file(self):
        self.stop_all()
        self.mode_fichier = True

        path, _ = QtWidgets.QFileDialog(self).getOpenFileName(self, "Sélectionner fichier", "", "Tous (*.*)")
        if not path:
            return

        try:
            df = pd.read_csv(path, sep="\t", decimal=",")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Lecture impossible :\n{e}")
            return

        if "N.A." in df.columns:
            df = df.drop(columns=["N.A."])

        df.rename(columns={"V1 (V)": "amplitude", "Tps (s)": "time"}, inplace=True)
        if "time" not in df.columns or "amplitude" not in df.columns:
            QtWidgets.QMessageBox.critical(self, "Erreur", "Colonnes attendues : 'Tps (s)' et 'V1 (V)'.")
            return

        self.donnees_x = df["time"].astype(float).tolist()
        self.donnees_y = df["amplitude"].astype(float).tolist()
        self.pointer = 0

        self.full_x.clear()
        self.full_y.clear()

        self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        self.plot_overview.show()
        self.curve_overview.setData(self.donnees_x, self.donnees_y)
        self.cursor_overview.setValue(self.donnees_x[0] if self.donnees_x else 0)

        self.timer_acq.start(self._acq_interval_ms())
        self.timer_plot.start(self.plot_refresh_ms)
        self.timer_detect.start(self.detect_refresh_ms)

    # ---------------------------------------------------------------------
    #cycles

    def acquisition_cycle(self):
        if self.en_pause:
            return

        if self.mode_fichier:
            start = self.pointer
            end = self.pointer + self.taille_lot
            if start >= len(self.donnees_x):
                self.timer_acq.stop()
                return

            x_batch = self.donnees_x[start:end]
            y_batch = self.donnees_y[start:end]
            self.pointer = end

            if len(x_batch) == 0 or len(y_batch) == 0:
                self.timer_acq.stop()
                return

            for t, v in zip(x_batch, y_batch):
                self.full_x.append(float(t))
                self.full_y.append(float(v))

            if self.pointer < len(self.donnees_x):
                self.cursor_overview.setValue(self.donnees_x[self.pointer])
            else:
                self.cursor_overview.setValue(self.donnees_x[-1])

        else:
            if not self.task:
                return
            try:
                chunk = self.task.read(number_of_samples_per_channel=self.taille_lot)
            except Exception as e:
                logger.exception("NI Error: %s", e)
                return

            if isinstance(chunk, float):
                chunk = [chunk]
            chunk = list(chunk)

            tlot = [self.current_time + i / self.fs for i in range(len(chunk))]
            self.current_time += len(chunk) / self.fs

            for t, v in zip(tlot, chunk):
                self.full_x.append(float(t))
                self.full_y.append(float(v))

            #Enregistrement
            if self.is_recording and self.recording_writer is not None:
                for t, v in zip(tlot, chunk):
                    self.recording_writer.writerow([t, v])
                try:
                    self.recording_file.flush()
                except Exception:
                    pass

        self.purge_old_data()
        self._recalculer_hold_steps()

    def affichage_cycle(self):
        if len(self.full_x) < 2:
            return

        #Récupère le buffer
        dx = np.fromiter(self.full_x, dtype=float)
        dy = np.fromiter(self.full_y, dtype=float)

        #Inversion
        if self.btn_inverted.isChecked():
            dy = -dy

        #Filtres
        cutoff1 = self.spin_cutoff1.value()
        cutoff2 = self.spin_cutoff2.value()
        ordre = self.spin_order.value()
        fenetre = self.spin_fenetre_moyenne.value()

        if self.chk_filtre_passe_bas.isChecked():
            dy = self.filtre_passe_bas(dy, cutoff1, self.fs, ordre)
        if self.chk_filtre_passe_haut.isChecked():
            dy = self.filtre_passe_haut(dy, cutoff2, self.fs, ordre)
        if self.chk_filtre_moyenne.isChecked():
            dy = self.filtre_moyenne_glissante(dy, fenetre)

        #Courbe
        self.curve.setData(dx, dy)

        #Mode d'affichage
        if self.current_display_mode == "simple":
            if self.simple_hold_counter > 0 and self.simple_hold_center is not None:
                center = self.simple_hold_center
                self.simple_hold_counter -= 1
            else:
                center = self.simple_last_peak_time if self.simple_last_peak_time is not None else float(dx[-1])

            half = self.simple_window_s / 2.0
            self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            self.plot.setXRange(center - half, center + half, padding=0)

            pos1 = center - half + 0.25 * self.simple_window_s
            pos2 = center - half + 0.75 * self.simple_window_s
            self.l1.setValue(pos1)
            self.l2.setValue(pos2)

        else:
            if len(dx) > 1:
                self.plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
                xmax = float(dx[-1])
                xmin = max(float(dx[0]), xmax - self.window_scroll_s)
                self.plot.setXRange(xmin, xmax, padding=0)

        #Stats et BPM
        self.actualiser_stats()
        self.update_bpm()

    def purge_old_data(self):
        pass

    #---------------------------------------------------------------------
    #Enregistrement

    def toggle_recording(self, checked):
        if checked:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.mode_fichier:
            QtWidgets.QMessageBox.warning(
                self,
                "Enregistrement impossible",
                "L'enregistrement n'est possible qu'en mode acquisition en direct.",
            )
            self.btn_record.setChecked(False)
            return

        nom_fichier, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer l'acquisition", "", "CSV (*.csv)")
        if not nom_fichier:
            self.btn_record.setChecked(False)
            return

        try:
            self.recording_file = open(nom_fichier, "w", newline="", encoding="utf-8")
            self.recording_writer = csv.writer(self.recording_file, delimiter=";")
            self.recording_writer.writerow(["Tps (s)", "V1 (V)"])
            self.is_recording = True
            self.btn_record.setText("Arrêter l'enregistrement")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur d'enregistrement", str(e))
            self.btn_record.setChecked(False)
            self.recording_file = None
            self.recording_writer = None
            self.is_recording = False

    def stop_recording(self):
        if self.is_recording:
            try:
                if self.recording_file:
                    self.recording_file.close()
            except Exception:
                pass

        self.is_recording = False
        self.recording_file = None
        self.recording_writer = None
        self.btn_record.setText("Enregistrer")
        self.btn_record.setChecked(False)

    #---------------------------------------------------------------------
    #Mesures et stats

    def actualiser_mesure(self):
        p1, p2 = float(self.l1.value()), float(self.l2.value())
        self.lbl_dt.setText(f"Delta temps: {abs(p2 - p1):.3f} s")
        self.actualiser_stats()

    def actualiser_stats(self):
        data = self.curve.getData()
        if not data or len(data[0]) == 0:
            self.lbl_stat.setText("Amplitude stats: N/A")
            return

        x_arr = np.array(data[0], dtype=float)
        y_arr = np.array(data[1], dtype=float)

        lo, hi = sorted((float(self.l1.value()), float(self.l2.value())))
        mask = (x_arr >= lo) & (x_arr <= hi)

        if np.any(mask):
            sub = y_arr[mask]
            self.lbl_stat.setText(
                f"min {sub.min():.2f} V, max {sub.max():.2f} V, moy {sub.mean():.2f} V | "
                f"amp curseur {float(self.lamp.value()):.2f} V"
            )
        else:
            self.lbl_stat.setText("Amplitude stats: N/A")

    # ---------------------------------------------------------------------
    #Détection pics

    def detect_peaks(self):
        """
        Détection des pics sur une fenêtre récente (detect_window_s).
        Met à jour :
          - self.peaks (ScatterPlotItem)
          - self.simple_last_peak_time et hold
          - self.lbl_peak
        """
        data = self.curve.getData()
        if not data or len(data[0]) < 2:
            self.peaks.setData([], [])
            self._pics_x = np.array([], dtype=float)
            self._pics_y = np.array([], dtype=float)
            return

        x = np.array(data[0], dtype=float)
        y = np.array(data[1], dtype=float)

        #fenêtre de détection sur les dernières secondes
        t_fin = float(x[-1])
        t_debut = max(float(x[0]), t_fin - self.detect_window_s)
        masque = (x >= t_debut) & (x <= t_fin)

        x_w = x[masque]
        y_w = y[masque]
        if y_w.size < 2:
            self.peaks.setData([], [])
            self._pics_x = np.array([], dtype=float)
            self._pics_y = np.array([], dtype=float)
            return

        try:
            r = ecg_detectors_modified.pan_tompkins_detector(y_w, self.fs)
            refined = ecg_detectors_modified.refine_peaks_on_raw_ecg(
                r, y_w, self.fs, min_amplitude=0.5
            )
            refined = np.asarray(refined, dtype=int)
            refined = refined[(refined >= 0) & (refined < y_w.size)]
        except Exception as e:
            logger.debug("Détection pics impossible: %s", e)
            refined = np.array([], dtype=int)

        px = x_w[refined] if refined.size else np.array([], dtype=float)
        py = y_w[refined] if refined.size else np.array([], dtype=float)

        self._pics_x = px
        self._pics_y = py
        self.peaks.setData(px, py)

        if px.size:
            dernier_t = float(px[-1])
            if self.simple_last_peak_time is None or dernier_t > self.simple_last_peak_time:
                self.simple_last_peak_time = dernier_t
                self.simple_hold_center = dernier_t
                self.simple_hold_counter = self.simple_hold_steps
                self.lbl_peak.setText(f"Dernier pic : {float(py[-1]):.3f} V")

    def update_bpm(self):
        px = self.peaks.getData()[0]
        if px is None or len(px) < 2:
            self.lbl_bpm.setText("BPM : --")
            return

        rr = np.diff(np.array(px, dtype=float))
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if rr.size == 0:
            self.lbl_bpm.setText("BPM : --")
            return

        self.lbl_bpm.setText(f"BPM : {60.0 / float(rr.mean()):.2f}")

    # ---------------------------------------------------------------------
    #Filtres

    def filtre_passe_bas(self, data, cutoff, fs, ordre=4):
        data = np.asarray(data, dtype=float)
        wn = cutoff / (0.5 * fs)
        if wn <= 0 or wn >= 1:
            return data
        b, a = butter(ordre, wn, btype="low")

        #filtfilt nécessite un signal assez long
        padlen = 3 * (max(len(a), len(b)) - 1)
        if data.size <= padlen:
            return data
        return filtfilt(b, a, data)

    def filtre_passe_haut(self, data, cutoff, fs, ordre=4):
        data = np.asarray(data, dtype=float)
        wn = cutoff / (0.5 * fs)
        if wn <= 0 or wn >= 1:
            return data
        b, a = butter(ordre, wn, btype="high")

        padlen = 3 * (max(len(a), len(b)) - 1)
        if data.size <= padlen:
            return data
        return filtfilt(b, a, data)

    def filtre_moyenne_glissante(self, signal, fenetre=5):
        signal = np.asarray(signal, dtype=float)
        if fenetre <= 1 or fenetre > len(signal):
            return signal
        kernel = np.ones(int(fenetre), dtype=float) / float(fenetre)
        return np.convolve(signal, kernel, mode="same")

    # ---------------------------------------------------------------------
    #Capture

    def capture_ecran(self):
        exporter = ImageExporter(self.plot.plotItem)
        nom_fichier, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer l'image", "", "PNG (*.png)")
        if nom_fichier:
            exporter.export(nom_fichier)

    # ---------------------------------------------------------------------
    #Utilitaires

    def _acq_interval_ms(self) -> int:
        return max(1, int(1000 * self.taille_lot / max(1, int(self.fs))))

    def _recalculer_hold_steps(self):
        pas_ms = self._acq_interval_ms()
        self.simple_hold_steps = max(1, self.simple_hold_ms // max(1, pas_ms))

    # ---------------------------------------------------------------------
    # Fermeture propre

    def closeEvent(self, event):
        self.stop_all()
        super().closeEvent(event)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = QtWidgets.QApplication(sys.argv)

    pg.setConfigOptions(antialias=True)

    win = VisualiseurSignal()
    win.resize(1100, 780)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

