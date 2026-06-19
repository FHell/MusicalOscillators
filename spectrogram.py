"""
Generate spectrogram of demo.wav and verify musical correctness of output.
Overlays expected MIDI note frequencies and reports a per-note pitch check.
"""

import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from scipy.signal import spectrogram as scipy_spectrogram

WAV_PATH = "examples/demo.wav"
OUT_PATH = "examples/spectrogram.png"

# --- Read WAV (16-bit PCM mono) ---
with open(WAV_PATH, "rb") as f:
    data = f.read()

i = 12
while data[i:i+4] != b"fmt ": i += 8 + struct.unpack_from("<I", data, i+4)[0]
samplerate = struct.unpack_from("<HHI", data, i+8)[2]
i = 12
while data[i:i+4] != b"data": i += 8 + struct.unpack_from("<I", data, i+4)[0]
data_size = struct.unpack_from("<I", data, i+4)[0]
pcm = np.frombuffer(data[i+8:i+8+data_size], dtype=np.int16).astype(np.float32) / 32768.0

print(f"WAV: {samplerate} Hz, {len(pcm)/samplerate:.2f}s")

# --- Spectrogram ---
nperseg = 2048
noverlap = nperseg * 3 // 4
f, t, Sxx = scipy_spectrogram(pcm, fs=samplerate, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

# Musical range
f_lo, f_hi = 150, 1100
f_mask = (f >= f_lo) & (f <= f_hi)

# --- Note schedule ---
def midi_hz(n): return 440.0 * 2**((n - 69) / 12.0)

arp_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
chord_pitches = [60, 64, 67, 72]
chord_start = 0.25 * len(arp_pitches)
chord_end = chord_start + 1.5

arp_notes = [(p, midi_hz(p), 0.25*i, 0.25*(i+1)) for i, p in enumerate(arp_pitches)]
chord_notes = [(p, midi_hz(p), chord_start, chord_end) for p in chord_pitches]
names = {60:"C4",62:"D4",64:"E4",65:"F4",67:"G4",69:"A4",71:"B4",72:"C5"}

# --- Figure: 2 rows ---
fig, (ax_spec, ax_energy) = plt.subplots(2, 1, figsize=(14, 9),
                                          gridspec_kw={"height_ratios": [3, 1]})

# ── Spectrogram panel ──
im = ax_spec.pcolormesh(t, f[f_mask], Sxx_db[f_mask], shading="auto", cmap="inferno",
                        vmin=np.percentile(Sxx_db[f_mask], 15),
                        vmax=Sxx_db[f_mask].max())

# Colour scheme: chord-overlap notes (C4,E4,G4,C5) in cyan, others in yellow
chord_set = set(chord_pitches)
for (p, hz, t0, t1) in arp_notes:
    color = "#00e5ff" if p in chord_set else "#ffee58"
    lw = 2.5
    ax_spec.plot([t0, t1], [hz, hz], color=color, linewidth=lw, alpha=0.9, solid_capstyle="round")
    ax_spec.annotate(names[p], xy=(t0, hz), fontsize=7, color=color,
                     va="bottom", ha="left", xytext=(2, 2), textcoords="offset points")
for (p, hz, t0, t1) in chord_notes:
    ax_spec.plot([t0, t1], [hz, hz], color="#00e5ff", linewidth=3.5, alpha=0.9,
                 linestyle="--", solid_capstyle="round")

# Mark arpeggio / chord boundary
ax_spec.axvline(chord_start, color="white", linewidth=1, linestyle=":", alpha=0.6)
ax_spec.text(chord_start + 0.05, f_hi * 0.95, "chord →", color="white", fontsize=8, va="top")

ax_spec.set_xlim(0, t[-1])
ax_spec.set_ylim(f_lo, f_hi)
ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
ax_spec.set_title("Spectrogram of demo.wav  |  yellow = arpeggio-only notes, cyan = chord-overlap notes",
                  fontsize=11)
ax_spec.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
fig.colorbar(im, ax=ax_spec, label="Power (dB)")

# ── Energy-per-note panel ──
# For each arpeggio note, plot power at its frequency in each 0.25s window
windows = [(0.25*j, 0.25*(j+1)) for j in range(len(arp_pitches))]
win_centers = [0.25*j + 0.125 for j in range(len(arp_pitches))]

for (p, hz, _, _) in arp_notes:
    powers = []
    for t0, t1 in windows:
        s0, s1 = int(t0 * samplerate), int(t1 * samplerate)
        seg = pcm[s0:s1]
        win_fn = np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(seg * win_fn)) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1.0 / samplerate)
        band = (freqs >= hz * 0.95) & (freqs <= hz * 1.05)
        powers.append(spec[band].sum() if band.any() else 0.0)
    powers = np.array(powers)
    powers /= max(powers.max(), 1e-12)
    color = "#00e5ff" if p in chord_set else "#ffee58"
    ax_energy.plot(win_centers, powers, marker="o", markersize=4, color=color,
                   label=names[p], alpha=0.85, linewidth=1.5)

ax_energy.set_xlim(0, t[-1])
ax_energy.set_ylim(0, 1.05)
ax_energy.set_xlabel("Time (s) — window center", fontsize=11)
ax_energy.set_ylabel("Relative power", fontsize=11)
ax_energy.set_title("Normalized power at each note's expected frequency across arpeggio windows\n"
                    "(ideally each line peaks in its own window)", fontsize=10)
ax_energy.legend(ncol=8, fontsize=8, loc="upper left")
ax_energy.axvline(chord_start, color="gray", linewidth=1, linestyle=":")

# Mark each note's own window with a triangle
for i, (p, hz, t0, t1) in enumerate(arp_notes):
    color = "#00e5ff" if p in chord_set else "#ffee58"
    s0, s1 = int(t0*samplerate), int(t1*samplerate)
    seg = pcm[s0:s1]
    wfn = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * wfn))**2
    freqs = np.fft.rfftfreq(len(seg), 1.0/samplerate)
    band = (freqs >= hz*0.95) & (freqs <= hz*1.05)
    powers_all = []
    for t00, t11 in windows:
        ss0, ss1 = int(t00*samplerate), int(t11*samplerate)
        sg = pcm[ss0:ss1]
        sp = np.abs(np.fft.rfft(sg * np.hanning(len(sg))))**2
        fr = np.fft.rfftfreq(len(sg), 1.0/samplerate)
        b = (fr >= hz*0.95) & (fr <= hz*1.05)
        powers_all.append(sp[b].sum() if b.any() else 0.0)
    powers_all = np.array(powers_all)
    powers_all /= max(powers_all.max(), 1e-12)
    own_power = powers_all[i]
    ax_energy.annotate("▲", xy=(win_centers[i], own_power),
                       fontsize=7, color=color, ha="center", va="bottom")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")

# --- Frequency check table ---
print("\nPer-note frequency check (FFT peak in ±30% band, full note window):")
print(f"{'Note':>5} {'Expected Hz':>12} {'Peak Hz':>10} {'Error':>8}  {'Energy peaks at own window?':>28}")
for i, (p, hz, t0, t1) in enumerate(arp_notes):
    s0, s1 = int(t0*samplerate), int(t1*samplerate)
    seg = pcm[s0:s1]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))**2
    freqs = np.fft.rfftfreq(len(seg), 1.0/samplerate)
    band = (freqs >= hz*0.70) & (freqs <= hz*1.30)
    peak_hz = freqs[band][np.argmax(spec[band])] if band.any() else 0
    err = abs(peak_hz - hz) / hz * 100

    # Check if power at hz peaks in own window
    powers_all = []
    for t00, t11 in windows:
        ss0, ss1 = int(t00*samplerate), int(t11*samplerate)
        sg = pcm[ss0:ss1]
        sp = np.abs(np.fft.rfft(sg * np.hanning(len(sg))))**2
        fr = np.fft.rfftfreq(len(sg), 1.0/samplerate)
        b = (fr >= hz*0.95) & (fr <= hz*1.05)
        powers_all.append(sp[b].sum() if b.any() else 0.0)
    powers_all = np.array(powers_all)
    own_rank = np.sum(powers_all > powers_all[i])  # 0 = strongest
    own_ok = "YES" if own_rank == 0 else f"NO (rank {own_rank+1}/{len(arp_pitches)})"

    print(f"{names[p]:>5} {hz:>12.1f} {peak_hz:>10.1f} {err:>7.1f}%  {own_ok:>28}")
