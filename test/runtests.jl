using MusicalOscillators
using Test

@testset "MusicalOscillators" begin

    @testset "VLQ encoding round-trip" begin
        for v in (0, 1, 127, 128, 255, 8192, 16383, 16384, 1_000_000)
            io = IOBuffer()
            MusicalOscillators.write_vlq!(io, v)
            bytes = take!(io)
            decoded, pos = MusicalOscillators.read_vlq(bytes, 1)
            @test decoded == v
            @test pos == length(bytes) + 1
        end
    end

    @testset "note_frequency" begin
        @test note_frequency(69) ≈ 440.0
        @test note_frequency(57) ≈ 220.0          # A3
        @test note_frequency(81) ≈ 880.0          # A5
        @test note_frequency(60) ≈ 261.6256 atol = 1e-3   # middle C
    end

    @testset "MIDI write/read round-trip" begin
        notes = [
            Note(60, 100, 0.0, 0.5, 0),
            Note(64, 90, 0.5, 0.5, 0),
            Note(67, 80, 1.0, 1.0, 0),
            Note(72, 110, 1.0, 1.0, 0),   # chord with previous note
        ]
        path = tempname() * ".mid"
        write_midi(path, notes; tempo = 500_000, division = 480)
        midi = read_midi(path)

        @test midi.format == 0
        @test length(midi.notes) == length(notes)

        got = sort(midi.notes; by = n -> (n.start, n.pitch))
        want = sort(notes; by = n -> (n.start, n.pitch))
        for (g, w) in zip(got, want)
            @test g.pitch == w.pitch
            @test g.velocity == w.velocity
            @test g.start ≈ w.start atol = 1e-3
            @test g.duration ≈ w.duration atol = 1e-3
        end
        rm(path; force = true)
    end

    @testset "envelope shape" begin
        notes = [Note(69, 127, 1.0, 1.0, 0)]
        cfg = OscillatorConfig(attack = 0.1, release = 0.2)
        p = MusicalOscillators.OscParams(notes, cfg)

        @test MusicalOscillators.envelope(p, 1, 0.5) == 0.0          # before onset
        @test MusicalOscillators.envelope(p, 1, 1.0) == 0.0          # at onset
        @test MusicalOscillators.envelope(p, 1, 1.05) ≈ 0.5 atol = 1e-6  # mid-attack
        @test MusicalOscillators.envelope(p, 1, 1.5) ≈ 1.0           # sustain
        @test MusicalOscillators.envelope(p, 1, 2.1) ≈ 0.5 atol = 1e-6  # mid-release
        @test MusicalOscillators.envelope(p, 1, 3.0) == 0.0          # fully released
    end

    @testset "synthesize basic properties" begin
        sr = 8000
        notes = [Note(69, 100, 0.0, 0.5, 0)]
        cfg = OscillatorConfig(tail = 0.1, release = 0.1)
        sig = synthesize(notes, sr, cfg)

        expected_len = round(Int, (0.5 + cfg.release + cfg.tail) * sr)
        @test length(sig) == expected_len
        @test all(isfinite, sig)
        @test maximum(abs, sig) ≤ 1.0
        @test maximum(abs, sig) ≈ 0.9 atol = 1e-6   # peak-normalised to gain
    end

    @testset "synthesize produces the right pitch" begin
        sr = 16000
        pitch = 69                  # A4 = 440 Hz
        notes = [Note(pitch, 110, 0.0, 1.0, 0)]
        cfg = OscillatorConfig(tail = 0.0, release = 0.05, mu = 1.0, coupling = 0.0)
        sig = synthesize(notes, sr, cfg)

        # Count zero crossings over the sustained portion to estimate frequency.
        a = round(Int, 0.2 * sr)
        b = round(Int, 0.9 * sr)
        seg = sig[a:b]
        crossings = 0
        for i in 2:length(seg)
            if (seg[i - 1] < 0) != (seg[i] < 0)
                crossings += 1
            end
        end
        est_freq = crossings / 2 / ((b - a) / sr)
        @test est_freq ≈ note_frequency(pitch) rtol = 0.1
    end

    @testset "empty input is silence" begin
        sig = synthesize(Note[], 8000)
        @test length(sig) == 1
        @test all(iszero, sig)
    end

    @testset "WAV write/read round-trip" begin
        sr = 8000
        signal = [0.0, 0.5, -0.5, 0.9, -0.9, 0.1]
        path = tempname() * ".wav"
        write_wav(path, signal, sr)
        back, got_sr = read_wav(path)

        @test got_sr == sr
        @test length(back) == length(signal)
        for (a, b) in zip(signal, back)
            @test a ≈ b atol = 1e-3
        end
        rm(path; force = true)
    end

    @testset "end-to-end midi_to_wav" begin
        notes = [
            Note(60, 100, 0.0, 0.4, 0),
            Note(64, 100, 0.4, 0.4, 0),
            Note(67, 100, 0.8, 0.6, 0),
        ]
        midi_path = tempname() * ".mid"
        wav_path = tempname() * ".wav"
        write_midi(midi_path, notes)
        out = midi_to_wav(midi_path, wav_path; samplerate = 16000)

        @test out == wav_path
        @test isfile(wav_path)
        sig, sr = read_wav(wav_path)
        @test sr == 16000
        @test !isempty(sig)
        @test all(isfinite, sig)
        @test maximum(abs, sig) ≤ 1.0

        rm(midi_path; force = true)
        rm(wav_path; force = true)
    end

end
