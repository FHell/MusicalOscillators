# Generate a short demo MIDI file and render it to WAV with the oscillator bank.
#
#   julia --project=. examples/demo.jl
#
# Produces `demo.mid` and `demo.wav` in the current directory.

using MusicalOscillators

# A little arpeggio that resolves into a C-major chord.
notes = Note[]
melody = [60, 62, 64, 65, 67, 69, 71, 72]   # C major scale
for (i, p) in enumerate(melody)
    push!(notes, Note(p, 90, 0.25 * (i - 1), 0.25, 0))
end
# final sustained chord
chord_start = 0.25 * length(melody)
for p in (60, 64, 67, 72)
    push!(notes, Note(p, 110, chord_start, 1.5, 0))
end

midi_path = joinpath(@__DIR__, "demo.mid")
wav_path = joinpath(@__DIR__, "demo.wav")

write_midi(midi_path, notes)
println("Wrote MIDI: ", midi_path)

config = OscillatorConfig(coupling = 0.03)   # mu=10, damp=5 from defaults
midi_to_wav(midi_path, wav_path; samplerate = 44100, config = config)
println("Wrote WAV:  ", wav_path)
