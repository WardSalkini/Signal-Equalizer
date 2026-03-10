import numpy as np
import pywt

# Simulate what the app does
signal = np.random.randn(3840)  # 30s at 128Hz
sr = 128.0
gains = [0.2, 1.0, 1.0, 1.0]  # Low Normal Sinus Rhythm gain

sliders = [
    {"name": "Normal Sinus Rhythm", "ranges": [[0.5, 5], [5, 15], [15, 40]]},
    {"name": "Atrial Fibrillation", "ranges": [[4, 9], [150, 250], [300, 500]]},
    {"name": "Ventricular Tachycardia", "ranges": [[40, 100], [100, 250]]},
    {"name": "Bradycardia", "ranges": [[0.1, 0.5], [0.5, 2]]}
]

wavelet = "db4"
level = min(pywt.dwt_max_level(len(signal), wavelet), 8)
print(f"Decomposition level: {level}")
print(f"Nyquist: {sr/2} Hz")

coeffs = pywt.wavedec(signal, wavelet, level=level)
print(f"Number of coeff arrays: {len(coeffs)}")
for lv in range(len(coeffs)):
    if lv == 0:
        lv_lo, lv_hi = 0, sr / (2 ** level)
    else:
        actual_level = level - lv + 1
        lv_lo = sr / (2 ** (actual_level + 1))
        lv_hi = sr / (2 ** actual_level)
    print(f"  coeffs[{lv}]: {len(coeffs[lv])} vals, freq: {lv_lo:.2f}-{lv_hi:.2f} Hz")

# Apply gains like the app does
eq_coeffs = [c.copy() for c in coeffs]
for i, slider in enumerate(sliders):
    g = gains[i]
    if g == 1.0:
        continue
    print(f"\nSlider: {slider['name']} gain={g}")
    for rng in slider.get("ranges", []):
        lo, hi = rng[0], rng[1]
        for lv in range(len(eq_coeffs)):
            if lv == 0:
                lv_lo, lv_hi = 0, sr / (2 ** level)
            else:
                actual_level = level - lv + 1
                lv_lo = sr / (2 ** (actual_level + 1))
                lv_hi = sr / (2 ** actual_level)
            if lv_hi > lo and lv_lo < hi:
                overlap = min(lv_hi, hi) - max(lv_lo, lo)
                level_width = lv_hi - lv_lo
                if level_width > 0:
                    fraction = overlap / level_width
                    blended = 1.0 + (g - 1.0) * fraction
                    eq_coeffs[lv] = eq_coeffs[lv] * blended
                    print(f"  [{lo}-{hi}Hz] -> lvl {lv} [{lv_lo:.2f}-{lv_hi:.2f}Hz] frac={fraction:.3f} gain={blended:.3f}")

output = pywt.waverec(eq_coeffs, wavelet)[:len(signal)]
diff = np.max(np.abs(signal - output))
rms = np.sqrt(np.mean((signal - output)**2))
print(f"\nMax diff: {diff:.6f}")
print(f"RMS diff: {rms:.6f}")
print(f"Signal changed: {diff > 0.001}")
