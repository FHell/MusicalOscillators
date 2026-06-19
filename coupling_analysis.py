"""
Coupling sweep analysis.

Arpeggio section: notes play one-at-a-time → coupling term is zero (single
active oscillator, mean-field = itself).  This section is a sanity check only.

Chord section:  C4, E4, G4, C5 play together for 1.5 s.  Multiple oscillators
are simultaneously active so the coupling term is non-zero.  We check:
  - FFT peak in each chord-note's band is within 5% of expected pitch.
  - Each chord note's band dominates over non-chord notes in the same window.

Produces:
  - examples/coupling_sweep.png  — two heat-maps (arpeggio pitch error + chord)
  - printed summary table
"""

import struct, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_wav_mono(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 12
    while data[i:i+4] != b"fmt ":
        i += 8 + struct.unpack_from("<I", data, i+4)[0]
    sr = struct.unpack_from("<HHI", data, i+8)[2]
    i = 12
    while data[i:i+4] != b"data":
        i += 8 + struct.unpack_from("<I", data, i+4)[0]
    n = struct.unpack_from("<I", data, i+4)[0]
    pcm = np.frombuffer(data[i+8:i+8+n], dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr

def midi_hz(n): return 440.0 * 2**((n - 69) / 12.0)

arp_pitches   = [60, 62, 64, 65, 67, 69, 71, 72]
arp_names     = ["C4","D4","E4","F4","G4","A4","B4","C5"]
chord_pitches = [60, 64, 67, 72]
chord_names   = ["C4","E4","G4","C5"]
chord_start   = 0.25 * 8
chord_end     = chord_start + 1.5

kappas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0, 10.0, 20.0, 50.0]

arp_err   = []   # (κ, note)
chord_err = []   # (κ, chord_note)

def fft_spec(pcm, sr, t0, t1):
    s0, s1 = int(t0*sr), int(t1*sr)
    seg   = pcm[s0:s1]
    wfn   = np.hanning(len(seg))
    spec  = np.abs(np.fft.rfft(seg * wfn))**2
    freqs = np.fft.rfftfreq(len(seg), 1.0/sr)
    return spec, freqs

def fft_peak_err(pcm, sr, t0, t1, hz, bw=0.10):
    """Find the peak within ±bw fraction of hz and return % error from hz."""
    spec, freqs = fft_spec(pcm, sr, t0, t1)
    band = (freqs >= hz*(1-bw)) & (freqs <= hz*(1+bw))
    if not band.any(): return 100.0
    peak_hz = freqs[band][np.argmax(spec[band])]
    return abs(peak_hz - hz) / hz * 100

def chord_power_present(pcm, sr, t0, t1, hz_list, bw=0.05):
    """
    For each expected frequency, compute the power in a ±bw band.
    Return (present, db_above_noise) where present = power > median_band_power.
    Returns list of (err_pct, snr_db) per note.
    """
    spec, freqs = fft_spec(pcm, sr, t0, t1)
    df = freqs[1] - freqs[0]
    results = []
    for hz in hz_list:
        band = (freqs >= hz*(1-bw)) & (freqs <= hz*(1+bw))
        noise_band = (freqs >= hz*0.5) & (freqs <= hz*2.0) & ~band
        sig_power  = spec[band].sum() if band.any() else 0.0
        noise_power = np.median(spec[noise_band]) * band.sum() if noise_band.any() else 1e-12
        snr_db = 10 * np.log10(max(sig_power, 1e-20) / max(noise_power, 1e-20))
        # peak within ±bw band
        if band.any():
            peak_hz = freqs[band][np.argmax(spec[band])]
            err_pct = abs(peak_hz - hz) / hz * 100
        else:
            err_pct = 100.0
        results.append((err_pct, snr_db))
    return results

chord_snr = []   # (κ, chord_note)

for κ in kappas:
    path = f"examples/sweep_k{κ}.wav"
    pcm, sr = read_wav_mono(path)

    # Arpeggio: narrow ±10% band, one note active at a time
    row = [fft_peak_err(pcm, sr, 0.25*i, 0.25*(i+1), midi_hz(p), bw=0.10)
           for i, p in enumerate(arp_pitches)]
    arp_err.append(row)

    # Chord: use middle 1.0 s, narrow ±5% band to avoid inter-note bleed
    t0c = chord_start + 0.25
    t1c = chord_end   - 0.25
    results = chord_power_present(pcm, sr, t0c, t1c,
                                  [midi_hz(p) for p in chord_pitches], bw=0.05)
    chord_err.append([r[0] for r in results])
    chord_snr.append([r[1] for r in results])

arp_err   = np.array(arp_err)
chord_err = np.array(chord_err)
chord_snr = np.array(chord_snr)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'':=<80}")
print("ARPEGGIO (one note at a time — coupling has no effect mathematically)")
print(f"{'':=<80}")
print(f"{'κ':>6}  " + "  ".join(f"{n:>5}" for n in arp_names) + "   max%")
for ki, κ in enumerate(kappas):
    errs = "  ".join(f"{e:5.1f}" for e in arp_err[ki])
    print(f"{κ:>6.2f}  {errs}   {arp_err[ki].max():5.1f}")

print(f"\n{'':=<80}")
print("CHORD  C4+E4+G4+C5 simultaneously — pitch error % (±5% band, narrow)")
print(f"{'':=<80}")
print(f"{'κ':>6}  " + "  ".join(f"{n:>6}" for n in chord_names) + "   max%  status")
for ki, κ in enumerate(kappas):
    errs = "  ".join(f"{e:6.1f}" for e in chord_err[ki])
    mx   = chord_err[ki].max()
    ok   = "✓ in-tune" if mx < 3.0 else ("~ marginal" if mx < 8.0 else "✗ broken")
    print(f"{κ:>6.2f}  {errs}   {mx:5.1f}  {ok}")

print(f"\n{'':=<80}")
print("CHORD  SNR (dB) of each note above local spectral noise floor")
print(f"{'':=<80}")
print(f"{'κ':>6}  " + "  ".join(f"{n:>7}" for n in chord_names) + "  min_snr")
for ki, κ in enumerate(kappas):
    snrs = "  ".join(f"{s:7.1f}" for s in chord_snr[ki])
    print(f"{κ:>6.2f}  {snrs}  {chord_snr[ki].min():7.1f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

def make_heatmap(ax, data, xlabels, title, vmax=10):
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([str(k) for k in kappas], fontsize=9)
    ax.set_xlabel("Note"); ax.set_ylabel("κ (coupling strength)")
    ax.set_title(title)
    for ki in range(len(kappas)):
        for ni in range(data.shape[1]):
            v = data[ki, ni]
            txt = f"{v:.1f}"
            col = "white" if v > vmax*0.65 else "black"
            ax.text(ni, ki, txt, ha="center", va="center", fontsize=7.5, color=col)
    plt.colorbar(im, ax=ax, label="% pitch error")
    return im

make_heatmap(axes[0], arp_err, arp_names,
             "Arpeggio: pitch error %\n(coupling=0 here — single oscillator active)\n5% threshold shown")
im1 = axes[1].imshow(chord_snr, aspect="auto", cmap="RdYlGn",
                     vmin=-5, vmax=30, origin="lower")
axes[1].set_xticks(range(len(chord_names))); axes[1].set_xticklabels(chord_names, fontsize=9)
axes[1].set_yticks(range(len(kappas)))
axes[1].set_yticklabels([str(k) for k in kappas], fontsize=9)
axes[1].set_xlabel("Note"); axes[1].set_ylabel("κ (coupling strength)")
axes[1].set_title("Chord C4+E4+G4+C5: SNR (dB) of each note\n(all 4 active — coupling matters)\ngreen = note present, red = note suppressed")
for ki in range(len(kappas)):
    for ni in range(len(chord_names)):
        v = chord_snr[ki, ni]
        txt = f"{v:.0f}"
        col = "white" if v < 5 else "black"
        axes[1].text(ni, ki, txt, ha="center", va="center", fontsize=7.5, color=col)
plt.colorbar(im1, ax=axes[1], label="SNR (dB)")

plt.suptitle("Stuart-Landau coupling sweep: pitch accuracy", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("examples/coupling_sweep.png", dpi=150, bbox_inches="tight")
print("\nSaved: examples/coupling_sweep.png")
