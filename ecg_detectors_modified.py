"""
A collection of ECG heartbeat detection algorithms implemented
in Python. Developed in conjunction with a new ECG database:
http://researchdata.gla.ac.uk/716/

Copyright (C) 2019-2023 Luis Howell & Bernd Porr
GPL GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

import numpy as np
import scipy.signal as signal
#NOTE : Cet algo a été modifié de l'original
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

def pan_tompkins_detector(unfiltered_ecg, fs):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.
    """
    maxQRSduration = 0.150  # sec
    N = int(maxQRSduration * fs)
    if len(unfiltered_ecg) < N:
        return []

    f1 = 5 / fs
    f2 = 15 / fs

    b, a = signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')
    filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

    diff = np.diff(filtered_ecg)
    squared = diff * diff

    mwa = MWA_cumulative(squared, N)
    mwa[:int(maxQRSduration * fs * 2)] = 0

    raw_peaks = panPeakDetect(mwa, fs)
    refined_peaks = refine_peaks_on_raw_ecg(raw_peaks, filtered_ecg, fs)
    return refined_peaks

def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(1, len(detection) - 1):
        if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
            peak = i
            peaks.append(i)

            if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                signal_peaks.append(peak)
                indexes.append(index)
                SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                if RR_missed != 0:
                    if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                        missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                        missed_section_peaks2 = []
                        for missed_peak in missed_section_peaks:
                            if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                missed_section_peaks2.append(missed_peak)

                        if len(missed_section_peaks2) > 0:
                            signal_missed = [detection[i] for i in missed_section_peaks2]
                            index_max = np.argmax(signal_missed)
                            missed_peak = missed_section_peaks2[index_max]
                            missed_peaks.append(missed_peak)
                            signal_peaks.append(signal_peaks[-1])
                            signal_peaks[-2] = missed_peak

            else:
                noise_peaks.append(peak)
                NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_I2 = 0.5 * threshold_I1

            if len(signal_peaks) > 8:
                RR = np.diff(signal_peaks[-9:])
                RR_ave = int(np.mean(RR))
                RR_missed = int(1.66 * RR_ave)

            index = index + 1

    signal_peaks.pop(0)
    return signal_peaks

def MWA_cumulative(input_array, window_size):
    n = len(input_array)
    if window_size > n:
        window_size = n
    if window_size < 1:
        window_size = 1
    ret = np.cumsum(input_array, dtype=float)
    if window_size < n:
        ret[window_size:] = ret[window_size:] - ret[:-window_size]
    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size
    return ret

# --------- AJOUT D'UNE FONCTION DE RECENTRAGE DES PICS

""" L'algorithme de pan tomkins utilise de nombreux filtres pour traiter le signal et détecter les ondes R.
En conséquence, l'indice de l'onde R détectée (après filtrage) peut différer de l'indice réel du signal brut.
On ne peut pas directement appliquer l'indice sortant de l'algorithme à notre signal brut.
C'est là que cette fonction intervient : dans une fenêtre glissante de 200ms autour de l'indice sortant de l'algorithme, 
on va chercher le maximum local du signal brut. C'est ce maximum local qui sera affiché. """
def refine_peaks_on_raw_ecg(peaks, raw_ecg, fs, window_ms=200, min_amplitude=0):
    half_window = int((window_ms / 1000.0) * fs / 2)
    refined_peaks = []

    for peak in peaks:
        start = max(0, peak - half_window)
        end = min(len(raw_ecg), peak + half_window)
        segment = raw_ecg[start:end]
        if len(segment) == 0:
            continue
        local_max = np.argmax(segment)
        amplitude = segment[local_max]
        if amplitude > min_amplitude:
            refined_peaks.append(start + local_max)
    return refined_peaks
