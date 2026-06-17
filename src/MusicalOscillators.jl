"""
    MusicalOscillators

Render a MIDI file to a WAV audio file by driving a high-dimensional, coupled
system of ordinary differential equations.

Each note becomes a pair of state dimensions in a Van-der-Pol-style oscillator
whose natural frequency is the note's pitch.  The MIDI note-on/note-off signals
modulate per-oscillator envelopes, all oscillators are weakly coupled through a
mean-field term, and the audio signal is the **average** of every oscillator's
position.

# Quick start
```julia
using MusicalOscillators
midi_to_wav("song.mid", "song.wav")
```

The package is fully self-contained (no external dependencies): it ships its own
Standard MIDI File reader/writer ([`read_midi`](@ref), [`write_midi`](@ref)) and
WAV reader/writer ([`read_wav`](@ref), [`write_wav`](@ref)).
"""
module MusicalOscillators

export Note, MidiData, OscillatorConfig
export read_midi, write_midi, parse_smf, note_frequency
export read_wav, write_wav
export synthesize, midi_to_wav

include("midi.jl")
include("ode.jl")
include("wav.jl")

"""
    midi_to_wav(midi_path, wav_path; samplerate = 44100,
                config = OscillatorConfig()) -> String

Read the MIDI file at `midi_path`, synthesize audio by integrating the coupled
oscillator ODE, and write the result as a 16-bit PCM mono WAV file to
`wav_path`.  Returns `wav_path`.
"""
function midi_to_wav(midi_path::AbstractString, wav_path::AbstractString;
                     samplerate::Integer = 44100,
                     config::OscillatorConfig = OscillatorConfig())
    midi = read_midi(midi_path)
    signal = synthesize(midi.notes, samplerate, config)
    write_wav(wav_path, signal, samplerate)
    return wav_path
end

end # module
