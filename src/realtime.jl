# Real-time Stuart-Landau oscillator bank for interactive use.
#
# Each of the N oscillators maps to one chromatic pitch (MIDI notes 60-71 by
# default).  Notes are switched on/off via activate_note!/deactivate_note!.
# The ODE is integrated with a simple Euler step (fast, good enough for audio).
#
# Coupling is an N×N matrix K; the mean-field coupling term in oscillator i is
#   sum_j  coupling_scale * K[i,j] * (x_j - x_i)
#   sum_j  coupling_scale * K[i,j] * (y_j - y_i)
#
# Exported: RealtimeSynth, activate_note!, deactivate_note!,
#           shuffle_coupling!, set_coupling!, generate_samples!

export RealtimeSynth, activate_note!, deactivate_note!,
       shuffle_coupling!, set_coupling!, generate_samples!

"""
    RealtimeSynth(pitches; samplerate, mu, damp, attack, release,
                  oversample, coupling_scale)

Real-time oscillator bank.  `pitches` is a vector of MIDI note numbers, one
per oscillator.
"""
mutable struct RealtimeSynth
    N::Int
    pitches::Vector{Int}
    omegas::Vector{Float64}     # 2π f_i
    u::Vector{Float64}          # ODE state [x1,y1,...,xN,yN]
    t::Float64
    A::Vector{Float64}          # current signed gain per oscillator
    K::Matrix{Float64}          # N×N coupling matrix (values in [0,1])
    coupling_scale::Float64
    mu::Float64
    damp::Float64
    attack::Float64
    release::Float64
    samplerate::Int
    oversample::Int
    lock::ReentrantLock
end

function RealtimeSynth(pitches::AbstractVector{<:Integer};
                       samplerate::Int     = 44100,
                       mu::Float64         = 10.0,
                       damp::Float64       = 5.0,
                       attack::Float64     = 0.02,
                       release::Float64    = 0.15,
                       oversample::Int     = 2,
                       coupling_scale::Float64 = 0.02)
    N = length(pitches)
    omegas = [2π * note_frequency(p) for p in pitches]
    K = zeros(Float64, N, N)
    RealtimeSynth(N, collect(Int, pitches), omegas,
                  zeros(Float64, 2N), 0.0,
                  fill(-damp, N), K, coupling_scale,
                  mu, damp, attack, release,
                  samplerate, oversample, ReentrantLock())
end

"""Trigger note-on for oscillator index `i` (1-based) at velocity `vel` ∈ [0,1]."""
function activate_note!(s::RealtimeSynth, i::Int, vel::Float64 = 1.0)
    lock(s.lock) do
        A_on = clamp(vel, 0.0, 1.0) * s.mu
        s.A[i] = A_on
        kick = sqrt(max(A_on, 0.0)) * 0.5
        s.u[2i - 1] = kick
    end
end

"""Release note for oscillator index `i` — let it ramp down via the release envelope."""
function deactivate_note!(s::RealtimeSynth, i::Int)
    lock(s.lock) do
        s.A[i] = -s.damp
    end
end

"""Randomise the coupling matrix K with values in [0, 1]."""
function shuffle_coupling!(s::RealtimeSynth)
    lock(s.lock) do
        rand!(s.K)
        # zero diagonal (self-coupling has no effect but keep it clean)
        for i in 1:s.N; s.K[i,i] = 0.0; end
    end
end

"""Set K[i,j] to `val` (clamped to [0,1])."""
function set_coupling!(s::RealtimeSynth, i::Int, j::Int, val::Float64)
    lock(s.lock) do
        s.K[i, j] = clamp(val, 0.0, 1.0)
    end
end

"""
    generate_samples!(s, buf)

Fill `buf` with the next `length(buf)` audio samples (Float64 in [-1,1]).
Thread-safe; acquires `s.lock` for the full call.
"""
function generate_samples!(s::RealtimeSynth, buf::AbstractVector{Float64})
    lock(s.lock) do
        N  = s.N
        os = max(s.oversample, 1)
        dt = 1.0 / (s.samplerate * os)
        u  = s.u
        A  = s.A
        K  = s.K
        κs = s.coupling_scale
        omegas = s.omegas
        nsamples = length(buf)

        for k in 1:nsamples
            for _ in 1:os
                # Euler step (low-latency, sufficient for audio rates)
                for i in 1:N
                    xi  = u[2i - 1]
                    yi  = u[2i]
                    ri2 = xi * xi + yi * yi
                    Ai  = A[i]
                    ωi  = omegas[i]
                    cx  = 0.0; cy = 0.0
                    @inbounds for j in 1:N
                        cx += K[i, j] * (u[2j - 1] - xi)
                        cy += K[i, j] * (u[2j]     - yi)
                    end
                    u[2i - 1] += dt * (xi * (Ai - ri2) - ωi * yi  + κs * cx)
                    u[2i]     += dt * (yi * (Ai - ri2) + ωi * xi  + κs * cy)
                end
                s.t += dt
            end
            sx = 0.0
            @inbounds for i in 1:N; sx += u[2i - 1]; end
            buf[k] = sx / N
        end
    end
    return buf
end
