"""
Signal Equalizer – ECG Abnormalities Mode
Flask backend with FFT/wavelet-based frequency equalization.
- wavelet and its silders what to do
- anither wired type of wavelet just to check  that the transformation is worrking
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json
import os
import io
import wave
import struct
from scipy import signal as scipy_signal

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

app = Flask(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────
store = {}  # { 'signal': [...], 'sr': 500, 'duration': ..., 'n_samples': ..., 'output': [...] }


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_csv_signal(text):
    """Read a single-channel or multi-column CSV. Returns (signal_1d, sr)."""
    lines = text.strip().split('\n')
    values = []
    sr = 500.0  # default
    for line in lines:
        parts = line.strip().split(',')
        try:
            vals = [float(p) for p in parts]
            values.append(vals[0] if len(vals) == 1 else vals[1])
        except ValueError:
            # check if header contains sr info
            if 'sr' in line.lower() or 'sample' in line.lower():
                for p in parts:
                    try:
                        sr = float(p)
                    except:
                        pass
    return np.array(values, dtype=np.float64), sr


def read_dat_signal(raw_bytes):
    """Read raw 16-bit DAT file as ECG signal."""
    sig = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
    sig = sig / (np.max(np.abs(sig)) + 1e-10)
    return sig, 500.0




def apply_fft_equalization(signal, sr, slider_settings, gains):
    """Apply equalization in frequency domain using FFT."""
    n = len(signal)
    fft_data = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Build per-frequency gain array
    gain_arr = np.ones(len(freqs))
    for i, slider in enumerate(slider_settings):
        g = gains[i] if i < len(gains) else 1.0
        for rng in slider.get('ranges', []):
            lo, hi = rng[0], rng[1]
            mask = (freqs >= lo) & (freqs <= hi)
            gain_arr[mask] = g

    fft_eq = fft_data * gain_arr
    output = np.fft.irfft(fft_eq, n=n)
    return output, freqs, np.abs(fft_data), np.abs(fft_eq), gain_arr


def apply_wavelet_equalization(signal, sr, slider_settings, gains, wavelet='db4', level=None):
    """Apply equalization using wavelet transform."""
    if not HAS_PYWT:
        return signal, None, None, None, None, None

    if level is None:
        level = min(pywt.dwt_max_level(len(signal), wavelet), 8)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Map each slider's frequency ranges to wavelet levels
    # Approximate frequency range per level:
    # Level j covers ~ [sr/2^(j+1), sr/2^j] Hz
    eq_coeffs = [c.copy() for c in coeffs]

    # Compute input RMS magnitudes per level BEFORE equalization
    input_mags = [float(np.sqrt(np.mean(c ** 2))) for c in coeffs]

    for i, slider in enumerate(slider_settings):
        g = gains[i] if i < len(gains) else 1.0
        for rng in slider.get('ranges', []):
            lo, hi = rng[0], rng[1]
            for lv in range(len(coeffs)):
                if lv == 0:
                    # Approximation coefficients: 0 to sr/2^level
                    lv_lo, lv_hi = 0, sr / (2 ** level)
                else:
                    # Detail coefficients at level (level - lv + 1) from decomposition
                    actual_level = level - lv + 1
                    lv_lo = sr / (2 ** (actual_level + 1))
                    lv_hi = sr / (2 ** actual_level)
                # Check overlap between slider range and wavelet level range
                if lv_hi > lo and lv_lo < hi:
                    overlap = min(lv_hi, hi) - max(lv_lo, lo)
                    level_width = lv_hi - lv_lo
                    if level_width > 0:
                        fraction = overlap / level_width
                        # Blend gain: weighted between original and target
                        blended = 1.0 + (g - 1.0) * fraction
                        eq_coeffs[lv] = eq_coeffs[lv] * blended

    # Compute output RMS magnitudes per level AFTER equalization
    output_mags = [float(np.sqrt(np.mean(c ** 2))) for c in eq_coeffs]

    # Build level labels with frequency ranges
    level_labels = []
    for lv in range(len(coeffs)):
        if lv == 0:
            lv_lo, lv_hi = 0, sr / (2 ** level)
            level_labels.append(f"cA ({lv_lo:.1f}–{lv_hi:.1f} Hz)")
        else:
            actual_level = level - lv + 1
            lv_lo = sr / (2 ** (actual_level + 1))
            lv_hi = sr / (2 ** actual_level)
            level_labels.append(f"cD{actual_level} ({lv_lo:.1f}–{lv_hi:.1f} Hz)")

    output = pywt.waverec(eq_coeffs, wavelet)
    # Trim to original length
    output = output[:len(signal)]

    return output, level_labels, input_mags, output_mags, None, True


def compute_spectrogram(signal, sr, nperseg=256):
    """Compute spectrogram data."""
    nperseg = min(nperseg, len(signal))
    f, t, Sxx = scipy_signal.spectrogram(signal, fs=sr, nperseg=nperseg,
                                          noverlap=nperseg // 2)
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f.tolist(), t.tolist(), Sxx_db.tolist()


def signal_to_wav_bytes(signal, sr):
    """Convert signal array to WAV bytes for audio playback."""
    # Normalize to 16-bit range
    s = signal / (np.max(np.abs(signal)) + 1e-10)
    s = (s * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(s.tobytes())
    buf.seek(0)
    return buf


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload an ECG signal file."""
    f = request.files.get('file')
    if not f:
        return jsonify(success=False, error='No file provided')

    filename = f.filename.lower()
    try:
        if filename.endswith('.csv'):
            text = f.read().decode('utf-8', errors='ignore')
            sig, sr = read_csv_signal(text)
        elif filename.endswith('.dat'):
            raw = f.read()
            sig, sr = read_dat_signal(raw)
        else:
            return jsonify(success=False, error='Unsupported file type. Use CSV or DAT.')

        if len(sig) == 0:
            return jsonify(success=False, error='Empty signal')

        duration = len(sig) / sr
        t = np.arange(len(sig)) / sr

        store['signal'] = sig
        store['sr'] = sr
        store['duration'] = duration
        store['n_samples'] = len(sig)
        store['time'] = t
        store['output'] = sig.copy()

        return jsonify(
            success=True,
            sr=sr,
            duration=duration,
            n_samples=len(sig),
            signal=sig.tolist(),
            time=t.tolist()
        )
    except Exception as e:
        return jsonify(success=False, error=str(e))




@app.route('/api/equalize', methods=['POST'])
def equalize():
    """Apply equalization with current slider gains."""
    if 'signal' not in store:
        return jsonify(success=False, error='No signal loaded')

    data = request.get_json()
    gains = data.get('gains', [1, 1, 1, 1])
    transform = data.get('transform', 'fourier')
    settings_sliders = data.get('sliders', [])

    sig = store['signal']
    sr = store['sr']

    try:
        print(f"[EQ] transform={transform}, gains={gains}, HAS_PYWT={HAS_PYWT}")
        if transform == 'fourier':
            output, freqs, input_mag, output_mag, gain_arr = \
                apply_fft_equalization(sig, sr, settings_sliders, gains)
            is_wavelet = False
        else:
            output, level_labels, input_mag, output_mag, _, is_wavelet = \
                apply_wavelet_equalization(sig, sr, settings_sliders, gains, wavelet=transform)
        print(f"[EQ] success, output len={len(output)}, is_wavelet={is_wavelet}")

        store['output'] = output

        if is_wavelet:
            # Wavelet mode: return coefficient magnitudes per level
            return jsonify(
                success=True,
                output=output.tolist(),
                is_wavelet=True,
                level_labels=level_labels,
                input_magnitude=input_mag,
                output_magnitude=output_mag
            )
        else:
            # FFT mode: downsample frequency data for transfer
            step = max(1, len(freqs) // 2000) if freqs is not None else 1
            freq_list = freqs[::step].tolist() if freqs is not None else []
            in_mag_list = input_mag[::step].tolist() if input_mag is not None else []
            out_mag_list = output_mag[::step].tolist() if output_mag is not None else []

            return jsonify(
                success=True,
                output=output.tolist(),
                is_wavelet=False,
                frequencies=freq_list,
                input_magnitude=in_mag_list,
                output_magnitude=out_mag_list
            )
    except Exception as e:
        return jsonify(success=False, error=str(e))


@app.route('/api/spectrogram', methods=['GET'])
def get_spectrogram():
    """Return spectrogram data for input or output signal."""
    which = request.args.get('which', 'input')
    if 'signal' not in store:
        return jsonify(success=False, error='No signal loaded')

    sig = store['signal'] if which == 'input' else store.get('output', store['signal'])
    sr = store['sr']

    try:
        f, t, Sxx = compute_spectrogram(sig, sr)
        return jsonify(success=True, frequencies=f, times=t, magnitudes=Sxx)
    except Exception as e:
        return jsonify(success=False, error=str(e))


@app.route('/api/audio', methods=['GET'])
def get_audio():
    """Return WAV audio for input or output signal."""
    which = request.args.get('which', 'output')
    if 'signal' not in store:
        return jsonify(success=False, error='No signal loaded'), 400

    sig = store['signal'] if which == 'input' else store.get('output', store['signal'])
    sr = store['sr']
    buf = signal_to_wav_bytes(sig, sr)
    return send_file(buf, mimetype='audio/wav', as_attachment=False,
                     download_name=f'{which}_signal.wav')


@app.route('/api/settings/save', methods=['POST'])
def save_settings():
    """Save current settings to a JSON file."""
    data = request.get_json()
    name = data.get('name', 'custom_settings')
    settings = data.get('settings', {})
    os.makedirs('settings', exist_ok=True)
    path = os.path.join('settings', f'{name}.json')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)
    return jsonify(success=True, path=path)


@app.route('/api/settings/load', methods=['POST'])
def load_settings():
    """Load settings from uploaded JSON file."""
    f = request.files.get('file')
    if not f:
        return jsonify(success=False, error='No file')
    try:
        settings = json.load(f)
        return jsonify(success=True, settings=settings)
    except Exception as e:
        return jsonify(success=False, error=str(e))


@app.route('/api/settings/default', methods=['GET'])
def default_settings():
    """Return the default ECG mode settings."""
    path = os.path.join('settings', 'ecg_mode.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return jsonify(success=True, settings=json.load(f))
    return jsonify(success=False, error='Default settings not found')


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
