# Sweep coupling strength and write one WAV per value.
# Usage: julia --project=. coupling_sweep.jl
using MusicalOscillators

notes = Note[]
melody = [60, 62, 64, 65, 67, 69, 71, 72]
for (i, p) in enumerate(melody)
    push!(notes, Note(p, 90, 0.25 * (i - 1), 0.25, 0))
end
chord_start = 0.25 * length(melody)
for p in (60, 64, 67, 72)
    push!(notes, Note(p, 110, chord_start, 1.5, 0))
end

write_midi(joinpath(@__DIR__, "examples/demo.mid"), notes)

kappas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0, 10.0, 20.0, 50.0]

for κ in kappas
    cfg = OscillatorConfig(coupling = κ)
    sig = synthesize(notes, 44100, cfg)
    path = joinpath(@__DIR__, "examples/sweep_k$(κ).wav")
    write_wav(path, sig, 44100)
    println("κ=$(κ)  wrote $(path)")
end
