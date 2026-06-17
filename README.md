# MusicalOscillators.jl

Render MIDI files to audio by driving a **high-dimensional, coupled system of
ordinary differential equations**. Every note in the score contributes a couple
of dimensions to one big ODE; the dimensions are integrated together and then
**averaged** into a single audio waveform that is written out as a WAV file.

The package is fully self-contained — it has **no external dependencies** and
ships its own Standard MIDI File reader/writer and 16-bit PCM WAV reader/writer.

## How it works

Each MIDI note `i` is modelled as two state variables `(xᵢ, yᵢ)` forming a
Van-der-Pol-style self-sustaining oscillator tuned to the note's pitch:

```
dxᵢ/dt = yᵢ
dyᵢ/dt = μ (aᵢ(t)² − xᵢ²) yᵢ − ωᵢ² xᵢ + κ (x̄ − xᵢ)
```

where

- `ωᵢ = 2π·fᵢ` is the angular frequency of the note (`fᵢ` from the MIDI pitch,
  A4 = 440 Hz),
- `aᵢ(t)` is a velocity-scaled attack/release envelope — this is the MIDI signal
  *manipulating* the oscillator. While the note sounds, `aᵢ > 0` and the
  Van der Pol pumping term sustains a limit-cycle oscillation; once the note is
  released `aᵢ → 0` and the same term becomes damping, so the oscillator rings
  down naturally.
- `x̄` is the mean position across **all** oscillators. The `κ(x̄ − xᵢ)` term is a
  weak mean-field coupling that ties the otherwise-independent notes into a
  single genuinely higher-dimensional, coupled ODE.
- `μ` controls how nonlinear (harmonically rich) each oscillator is.

For a score with `N` notes this is a `2N`-dimensional ODE. It is integrated with
a fixed-step classical Runge–Kutta (RK4) method at (an oversampled multiple of)
the audio sample rate. The output audio sample at each step is the **average of
all positions**, `mean_i xᵢ`, which is then peak-normalised and written to WAV.

## Installation

```julia
pkg> add https://github.com/FHell/MusicalOscillators
```

or, working from a clone:

```julia
pkg> activate .
pkg> instantiate
```

## Usage

```julia
using MusicalOscillators

# The one-liner: MIDI file in, WAV file out.
midi_to_wav("song.mid", "song.wav")

# Tweak the synthesis.
cfg = OscillatorConfig(
    mu        = 4.0,    # more harmonics
    coupling  = 0.05,   # stronger inter-note coupling
    attack    = 0.005,
    release   = 0.25,
)
midi_to_wav("song.mid", "song.wav"; samplerate = 48000, config = cfg)
```

Lower-level building blocks are exported too:

```julia
midi   = read_midi("song.mid")          # -> MidiData (vector of Notes)
signal = synthesize(midi.notes, 44100)  # -> Vector{Float64} mono audio
write_wav("song.wav", signal, 44100)

# Build a score programmatically and write a MIDI file.
notes = [Note(60, 100, 0.0, 0.5, 0), Note(64, 100, 0.5, 0.5, 0)]
write_midi("scale.mid", notes)
```

### `Note` fields

| field      | meaning                                  |
| ---------- | ---------------------------------------- |
| `pitch`    | MIDI note number (0–127, 69 = A4)        |
| `velocity` | note-on velocity (1–127)                 |
| `start`    | onset time in seconds                    |
| `duration` | duration in seconds                      |
| `channel`  | MIDI channel (0–15)                      |

### `OscillatorConfig` options

| option       | default | meaning                                      |
| ------------ | ------- | -------------------------------------------- |
| `mu`         | `3.0`   | Van der Pol nonlinearity (harmonic richness) |
| `coupling`   | `0.02`  | mean-field coupling strength `κ`             |
| `attack`     | `0.01`  | attack time (s)                              |
| `release`    | `0.15`  | release time (s)                             |
| `oversample` | `2`     | ODE substeps per audio sample                |
| `tail`       | `0.3`   | extra render time after last note (s)        |
| `gain`       | `0.9`   | peak normalisation target                    |

## Example

```bash
julia --project=. examples/demo.jl
```

writes `examples/demo.mid` and `examples/demo.wav` (a C-major scale resolving
into a sustained chord).

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

MIT
