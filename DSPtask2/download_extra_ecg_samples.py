"""
Download additional ECG samples from PhysioNet – one new record per existing label
plus one brand-new label (supraventricular tachycardia).
Uses the same CSV format as download_ecg_samples.py.
"""
import urllib.request
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ecg_samples')
os.makedirs(OUT_DIR, exist_ok=True)

# New records – every one is different from the original downloads
SAMPLES = [
    {
        'label': 'normal_sinus_rhythm_2',
        'hea_url': 'https://physionet.org/files/nsrdb/1.0.0/16272.hea',
        'dat_url': 'https://physionet.org/files/nsrdb/1.0.0/16272.dat',
        'desc': 'Normal Sinus Rhythm (nsrdb record 16272)',
    },
    {
        'label': 'atrial_fibrillation_2',
        'hea_url': 'https://physionet.org/files/afdb/1.0.0/04043.hea',
        'dat_url': 'https://physionet.org/files/afdb/1.0.0/04043.dat',
        'desc': 'Atrial Fibrillation (afdb record 04043)',
    },
    {
        'label': 'ventricular_tachycardia_2',
        'hea_url': 'https://physionet.org/files/vfdb/1.0.0/422.hea',
        'dat_url': 'https://physionet.org/files/vfdb/1.0.0/422.dat',
        'desc': 'Ventricular Tachycardia (vfdb record 422)',
    },
    {
        'label': 'bradycardia_2',
        'hea_url': 'https://physionet.org/files/mitdb/1.0.0/231.hea',
        'dat_url': 'https://physionet.org/files/mitdb/1.0.0/231.dat',
        'desc': 'Bradycardia (mitdb record 231)',
    },
    # ── NEW LABEL ──
    {
        'label': 'supraventricular_tachycardia',
        'hea_url': 'https://physionet.org/files/svdb/1.0.0/800.hea',
        'dat_url': 'https://physionet.org/files/svdb/1.0.0/800.dat',
        'desc': 'Supraventricular Tachycardia (svdb record 800)',
    },
]


def parse_header(hea_text):
    """Parse a WFDB header file to extract sample rate, number of channels, and format."""
    lines = hea_text.strip().split('\n')
    parts = lines[0].split()
    n_channels = int(parts[1])
    sr = float(parts[2])
    fmt = 16
    for line in lines[1:]:
        if line.startswith('#'):
            continue
        sig_parts = line.split()
        if len(sig_parts) >= 2:
            try:
                fmt = int(sig_parts[1])
            except:
                pass
            break
    return n_channels, sr, fmt


def decode_212(raw, n_channels, max_samples):
    """Decode WFDB format 212 (12-bit packed) data."""
    signals = []
    idx = 0
    raw_bytes = bytearray(raw)
    for _ in range(max_samples):
        if idx + 3 * n_channels // 2 > len(raw_bytes):
            break
        frame = []
        for ch_pair in range(0, n_channels, 2):
            if idx + 3 > len(raw_bytes):
                break
            b0 = raw_bytes[idx]
            b1 = raw_bytes[idx + 1]
            b2 = raw_bytes[idx + 2]
            idx += 3
            s1 = b0 | ((b1 & 0x0F) << 8)
            if s1 >= 2048:
                s1 -= 4096
            s2 = (b1 >> 4) | (b2 << 4)
            if s2 >= 2048:
                s2 -= 4096
            frame.append(s1)
            if ch_pair + 1 < n_channels:
                frame.append(s2)
        if len(frame) >= 1:
            signals.append(frame[0])
    return np.array(signals, dtype=np.float64)


def download_ecg(sample_info, duration_sec=30):
    """Download a small chunk of ECG data and save as CSV."""
    label = sample_info['label']
    print(f"\n{'='*60}")
    print(f"Downloading: {sample_info['desc']}")

    # Download header
    print(f"  Fetching header...")
    try:
        req = urllib.request.Request(sample_info['hea_url'],
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            hea_text = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  ERROR fetching header: {e}")
        return False

    n_channels, sr, fmt = parse_header(hea_text)
    print(f"  Channels: {n_channels}, Sample Rate: {sr} Hz, Format: {fmt}")

    n_samples = int(sr * duration_sec)

    if fmt == 212:
        bytes_per_frame = int(n_channels * 1.5)
        total_bytes = int(n_samples * bytes_per_frame)
    else:
        total_bytes = n_samples * n_channels * 2

    print(f"  Downloading {total_bytes / 1024:.1f} KB ({duration_sec}s of data)...")

    try:
        req = urllib.request.Request(sample_info['dat_url'], headers={
            'User-Agent': 'Mozilla/5.0',
            'Range': f'bytes=0-{total_bytes - 1}'
        })
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"  ERROR fetching data: {e}")
        return False

    print(f"  Downloaded {len(raw)} bytes")

    if fmt == 212:
        signal = decode_212(raw, n_channels, n_samples)
    else:
        vals = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            n_complete = len(vals) // n_channels
            vals = vals[:n_complete * n_channels].reshape(-1, n_channels)
            signal = vals[:, 0].astype(np.float64)
        else:
            signal = vals.astype(np.float64)

    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    time = np.arange(len(signal)) / sr

    csv_path = os.path.join(OUT_DIR, f'{label}.csv')
    print(f"  Saving to {csv_path} ({len(signal)} samples)...")
    with open(csv_path, 'w') as f:
        f.write(f'time,amplitude,sr,{sr}\n')
        for i in range(len(signal)):
            f.write(f'{time[i]:.6f},{signal[i]:.8f}\n')

    print(f"  [OK] Saved: {label}.csv")
    return True


if __name__ == '__main__':
    print("Extra ECG Sample Downloader")
    print("=" * 60)
    success = 0
    for sample in SAMPLES:
        if download_ecg(sample, duration_sec=30):
            success += 1
    print(f"\n{'=' * 60}")
    print(f"Done! Downloaded {success}/{len(SAMPLES)} samples to: {OUT_DIR}")
