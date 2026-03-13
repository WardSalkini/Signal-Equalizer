"""
Signal Equalizer
Flask backend — dual-domain equalization: FFT + optimal wavelet per mode.
Supports: CSV, DAT, WAV, MP3
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json, os, io, wave, struct, subprocess, tempfile
from scipy import signal as scipy_signal
from scipy.io import wavfile as scipy_wavfile

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

app = Flask(__name__)
store = {}

# ── Optimal wavelet per mode ───────────────────────────────────────────────────
OPTIMAL_WAVELET = {
    'ecg':    'sym5',   
    'music':  'db4',    
    'voices': 'haar',   
}

# ── Signal readers ─────────────────────────────────────────────────────────────

def read_csv_signal(text):
    lines = text.strip().split('\n')
    values, sr = [], 500.0
    for line in lines:
        parts = line.strip().split(',')
        try:
            vals = [float(p) for p in parts]
            values.append(vals[0] if len(vals) == 1 else vals[1])
        except ValueError:
            if 'sr' in line.lower() or 'sample' in line.lower():
                for p in parts:
                    try: sr = float(p)
                    except: pass
    return np.array(values, dtype=np.float64), sr

def read_dat_signal(raw_bytes):
    sig = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
    sig = sig / (np.max(np.abs(sig)) + 1e-10)
    return sig, 500.0

def read_wav_signal(raw_bytes):
    buf = io.BytesIO(raw_bytes)
    sr, data = scipy_wavfile.read(buf)
    dtype_map = {
        'int16':   lambda d: d.astype(np.float64) / 32768.0,
        'int32':   lambda d: d.astype(np.float64) / 2147483648.0,
        'uint8':   lambda d: (d.astype(np.float64) - 128.0) / 128.0,
        'float32': lambda d: d.astype(np.float64),
        'float64': lambda d: d.astype(np.float64),
    }
    conv = dtype_map.get(data.dtype.name, lambda d: d.astype(np.float64) / (np.max(np.abs(d)) + 1e-10))
    sig = conv(data)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    return sig, float(sr)

def find_ffmpeg():
    import shutil
    path = shutil.which('ffmpeg')
    if path: return path
    for p in [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        os.path.join(os.environ.get('LOCALAPPDATA',''), 'ffmpeg','bin','ffmpeg.exe'),
        os.path.join(os.environ.get('USERPROFILE',''),  'ffmpeg','bin','ffmpeg.exe'),
    ]:
        if p and os.path.isfile(p): return p
    return None

def read_mp3_signal(raw_bytes):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it and add to PATH: https://ffmpeg.org/download.html")
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp.write(raw_bytes); tmp_in = tmp.name
    tmp_out = tmp_in.replace('.mp3', '_out.wav')
    try:
        r = subprocess.run([ffmpeg,'-y','-i',tmp_in,'-ac','1','-ar','44100','-f','wav',tmp_out],
                           capture_output=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {r.stderr.decode('utf-8','ignore')}")
        with open(tmp_out,'rb') as f: wav = f.read()
        return read_wav_signal(wav)
    finally:
        for p in [tmp_in, tmp_out]:
            try: os.unlink(p)
            except: pass

# ── FFT equalization ───────────────────────────────────────────────────────────

def apply_fft_equalization(signal, sr, sliders, gains):
    n = len(signal)
    fft_data = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(n, d=1.0/sr)
    gain_arr = np.ones(len(freqs))
    for i, s in enumerate(sliders):
        g = gains[i] if i < len(gains) else 1.0
        if g == 1.0:
            continue  # neutral — skip, no effect on overlapping bins
        for lo, hi in s.get('ranges', []):
            mask = (freqs >= lo) & (freqs <= hi)
            # Multiply instead of overwrite — overlapping ranges compound correctly
            gain_arr[mask] *= g
    fft_eq = fft_data * gain_arr
    output = np.fft.irfft(fft_eq, n=n)
    return output, freqs, np.abs(fft_data), np.abs(fft_eq)

# ── Wavelet equalization ───────────────────────────────────────────────────────

def get_level_freq_range(lv, level, sr):
    if lv == 0:
        return 0.0, sr / (2 ** level)
    actual = level - lv + 1
    return sr / (2 ** (actual + 1)), sr / (2 ** actual)

def apply_wavelet_equalization(signal, sr, sliders, gains, wavelet):
    if not HAS_PYWT:
        return signal, [], [], [], []

    level = min(pywt.dwt_max_level(len(signal), wavelet), 8)
    coeffs    = pywt.wavedec(signal, wavelet, level=level)
    eq_coeffs = [c.copy() for c in coeffs]
    input_mags = [float(np.sqrt(np.mean(c**2))) for c in coeffs]

    component_level_map = []
    for i, s in enumerate(sliders):
        g = gains[i] if i < len(gains) else 1.0
        target_levels = s.get('wavelet_levels', [])
        for lv in target_levels:
            if 0 <= lv < len(eq_coeffs):
                eq_coeffs[lv] = eq_coeffs[lv] * g
        component_level_map.append({
            'color':  s.get('color', '#00e5ff'),
            'name':   s.get('name', f'Component {i+1}'),
            'levels': [lv for lv in target_levels if 0 <= lv < len(coeffs)]
        })

    output_mags = [float(np.sqrt(np.mean(c**2))) for c in eq_coeffs]

    level_labels = []
    for lv in range(len(coeffs)):
        lo, hi = get_level_freq_range(lv, level, sr)
        if lv == 0:
            level_labels.append(f"cA {lo:.1f}–{hi:.1f}Hz")
        else:
            actual = level - lv + 1
            level_labels.append(f"cD{actual} {lo:.1f}–{hi:.1f}Hz")

    output = pywt.waverec(eq_coeffs, wavelet)[:len(signal)]
    return output, level_labels, input_mags, output_mags, component_level_map


def compute_spectrogram(signal, sr, nperseg=256):
    nperseg = min(nperseg, len(signal))
    f, t, Sxx = scipy_signal.spectrogram(signal, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    return f.tolist(), t.tolist(), (10*np.log10(Sxx+1e-10)).tolist()

def signal_to_wav_bytes(signal, sr):
    s = signal / (np.max(np.abs(signal)) + 1e-10)
    s = (s * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(int(sr)); wf.writeframes(s.tobytes())
    buf.seek(0)
    return buf

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if not f: return jsonify(success=False, error='No file provided')
    fname = f.filename.lower()
    try:
        if   fname.endswith('.csv'): sig, sr = read_csv_signal(f.read().decode('utf-8','ignore'))
        elif fname.endswith('.dat'): sig, sr = read_dat_signal(f.read())
        elif fname.endswith('.wav'): sig, sr = read_wav_signal(f.read())
        elif fname.endswith('.mp3'): sig, sr = read_mp3_signal(f.read())
        else: return jsonify(success=False, error='Unsupported file type. Use CSV, DAT, WAV, or MP3.')
        if len(sig) == 0: return jsonify(success=False, error='Empty signal')
        MAX = int(sr * 60)
        if len(sig) > MAX: sig = sig[:MAX]
        t = np.arange(len(sig)) / sr
        
        # Initialize independent outputs
        store.update(signal=sig, sr=sr, duration=len(sig)/sr,
                     n_samples=len(sig), time=t, 
                     fft_output=sig.copy(), wav_output=sig.copy())
                     
        return jsonify(success=True, sr=sr, duration=store['duration'],
                       n_samples=len(sig), signal=sig.tolist(), time=t.tolist())
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))


@app.route('/api/equalize', methods=['POST'])
def equalize():
    """Independent FFT equalization"""
    if 'signal' not in store:
        return jsonify(success=False, error='No signal loaded')
    data    = request.get_json()
    gains   = data.get('gains', [])
    sliders = data.get('sliders', [])
    sig, sr = store['signal'], store['sr']
    try:
        output, freqs, in_mag, out_mag = apply_fft_equalization(sig, sr, sliders, gains)
        store['fft_output'] = output
        
        step = max(1, len(freqs)//2000)
        return jsonify(
            success=True,
            output=output.tolist(),
            frequencies=freqs[::step].tolist(),
            input_magnitude=in_mag[::step].tolist(),
            output_magnitude=out_mag[::step].tolist()
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/wavelet_equalize', methods=['POST'])
def wavelet_equalize():
    """Independent Wavelet equalization with FFT output for visualization"""
    if 'signal' not in store:
        return jsonify(success=False, error='No signal loaded')
    data    = request.get_json()
    gains   = data.get('gains', [])
    sliders = data.get('sliders', [])
    mode    = data.get('mode', 'ecg')
    
    # Process from the raw signal to ensure independence
    base_sig = store['signal']
    sr       = store['sr']
    wavelet  = OPTIMAL_WAVELET.get(mode, 'db4')
    try:
        output, level_labels, in_mags, out_mags, comp_map = \
            apply_wavelet_equalization(base_sig, sr, sliders, gains, wavelet)
        store['wav_output'] = output
        
        # Compute FFT of base_sig and wavelet output for the new viewer
        n = len(base_sig)
        freqs = np.fft.rfftfreq(n, d=1.0/sr)
        fft_in = np.abs(np.fft.rfft(base_sig))
        fft_out = np.abs(np.fft.rfft(output))
        step = max(1, len(freqs)//2000)

        return jsonify(
            success=True,
            output=output.tolist(),
            wavelet=wavelet,
            level_labels=level_labels,
            input_magnitude=in_mags,
            output_magnitude=out_mags,
            component_map=comp_map,
            # Pass the frequency domain data back to the frontend
            frequencies=freqs[::step].tolist(),
            fft_in_mag=fft_in[::step].tolist(),
            fft_out_mag=fft_out[::step].tolist()
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))


@app.route('/api/spectrogram', methods=['GET'])
def get_spectrogram():
    which = request.args.get('which','input')
    if 'signal' not in store: return jsonify(success=False, error='No signal loaded')
    
    if which == 'fft':
        sig = store.get('fft_output', store['signal'])
    elif which == 'wav':
        sig = store.get('wav_output', store['signal'])
    else:
        sig = store['signal']
        
    try:
        f, t, Sxx = compute_spectrogram(sig, store['sr'])
        return jsonify(success=True, frequencies=f, times=t, magnitudes=Sxx)
    except Exception as e:
        return jsonify(success=False, error=str(e))


@app.route('/api/audio', methods=['GET'])
def get_audio():
    which = request.args.get('which','fft')
    if 'signal' not in store: return jsonify(success=False, error='No signal loaded'), 400
    
    if which == 'fft':
        sig = store.get('fft_output', store['signal'])
    elif which == 'wav':
        sig = store.get('wav_output', store['signal'])
    else:
        sig = store['signal']
        
    return send_file(signal_to_wav_bytes(sig, store['sr']), mimetype='audio/wav',
                     as_attachment=False, download_name=f'{which}_signal.wav')


@app.route('/api/settings/save', methods=['POST'])
def save_settings():
    data = request.get_json()
    os.makedirs('settings', exist_ok=True)
    path = os.path.join('settings', f"{data.get('name','custom')}.json")
    with open(path,'w') as f: json.dump(data.get('settings',{}), f, indent=2)
    return jsonify(success=True, path=path)

@app.route('/api/settings/load', methods=['POST'])
def load_settings():
    f = request.files.get('file')
    if not f: return jsonify(success=False, error='No file')
    try: return jsonify(success=True, settings=json.load(f))
    except Exception as e: return jsonify(success=False, error=str(e))

@app.route('/api/settings/default', methods=['GET'])
def default_settings():
    path = os.path.join('settings','ecg_mode.json')
    if os.path.exists(path):
        with open(path,'r') as f: return jsonify(success=True, settings=json.load(f))
    return jsonify(success=False, error='Default settings not found')

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)