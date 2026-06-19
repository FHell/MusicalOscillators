# The synthesis core.
#
# Each MIDI note is modelled as a **Stuart-Landau oscillator** — a complex
# normal-form oscillator whose amplitude tracks the MIDI envelope directly:
#
#     dv_i/dt = v_i (iΩ_i − |v_i|² + A_i(t)) + κ (v̄ − v_i)
#
# Writing v_i = x_i + i y_i this becomes the real-valued system
#
#     dx_i/dt =  x_i (A_i(t) − r_i²) − Ω_i y_i  +  κ (x̄ − x_i)
#     dy_i/dt =  y_i (A_i(t) − r_i²) + Ω_i x_i  +  κ (ȳ − y_i)
#
# where
#   * Ω_i      is the angular frequency of note i,
#   * r_i²     = x_i² + y_i²,
#   * A_i(t)   is a **signed** MIDI-derived gain:
#                > 0 during the note  →  stable limit cycle at r_i = √A_i,
#                < 0 between notes    →  origin is stable, oscillator decays.
#   * v̄        is the mean phasor (mean-field coupling),
#   * κ        is the coupling strength.
#
# Inactive oscillators are held near zero by the negative gain −damp.
# Each note gets an amplitude **kick** at its scheduled start so the
# oscillator reaches the limit cycle within the note duration rather
# than growing slowly from numerical noise.
#
# The audio output is the average of every note's real part, mean_i x_i.

"""
    OscillatorConfig(; kwargs...)

Parameters controlling the oscillator bank.

- `mu`          : peak gain A_max; sustained amplitude ≈ √(vel · mu) (default 10.0).
- `coupling`    : mean-field coupling strength κ between oscillators (default 0.02).
- `damp`        : inactive damping rate; A_i = −damp when note is off (default 5.0).
- `attack`      : note attack time in seconds (default 0.01).
- `release`     : note release time in seconds (default 0.15).
- `oversample`  : ODE substeps per audio sample for stability (default 2).
- `tail`        : extra seconds rendered after the last note ends (default 0.3).
- `gain`        : peak normalisation target in [0, 1] (default 0.9).
"""
Base.@kwdef struct OscillatorConfig
    mu::Float64       = 10.0
    coupling::Float64 = 0.02
    damp::Float64     = 5.0
    attack::Float64   = 0.01
    release::Float64  = 0.15
    oversample::Int   = 2
    tail::Float64     = 0.3
    gain::Float64     = 0.9
end

# Flattened, integration-ready parameters.
struct OscParams
    N::Int
    omega::Vector{Float64}     # Ω_i = 2π f_i  (angular frequency, not squared)
    starts::Vector{Float64}
    durs::Vector{Float64}
    vels::Vector{Float64}      # normalised to [0, 1]
    mu::Float64                # A_max (peak gain)
    kappa::Float64
    damp::Float64
    attack::Float64
    release::Float64
end

function OscParams(notes::AbstractVector{Note}, cfg::OscillatorConfig)
    N = length(notes)
    omega  = Vector{Float64}(undef, N)
    starts = Vector{Float64}(undef, N)
    durs   = Vector{Float64}(undef, N)
    vels   = Vector{Float64}(undef, N)
    for (i, n) in enumerate(notes)
        omega[i]  = 2π * note_frequency(n.pitch)
        starts[i] = n.start
        durs[i]   = n.duration
        vels[i]   = clamp(n.velocity / 127, 0.0, 1.0)
    end
    return OscParams(N, omega, starts, durs, vels, cfg.mu, cfg.coupling,
                     cfg.damp, cfg.attack, cfg.release)
end

# Signed Stuart-Landau gain A_i(t) for note `i` at time `t`.
#
# Returns −damp when the note is inactive (oscillator decays to zero),
# ramps linearly to vel·mu during attack, holds during sustain, and
# ramps back to −damp over the release window.
@inline function envelope(p::OscParams, i::Int, t::Float64)
    s     = p.starts[i]
    d     = p.durs[i]
    e     = s + d
    v     = p.vels[i]
    a_t   = p.attack
    A_on  = v * p.mu       # positive sustain level
    A_off = -p.damp        # negative inactive level

    if t < s
        return A_off
    elseif t < s + a_t
        α = (t - s) / a_t
        return A_off + α * (A_on - A_off)
    elseif t <= e
        return A_on
    else
        # level at the moment of note-off (may not have reached A_on if attack > dur)
        held = d < a_t ? A_off + (d / a_t) * (A_on - A_off) : A_on
        rt = t - e
        rt < p.release ? held + (rt / p.release) * (A_off - held) : A_off
    end
end

# Preallocated scratch space for the RK4 integrator.
struct Workspace
    k1::Vector{Float64}
    k2::Vector{Float64}
    k3::Vector{Float64}
    k4::Vector{Float64}
    tmp::Vector{Float64}
end
Workspace(n::Int) = Workspace(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))

# In-place derivative of the full coupled Stuart-Landau system.
function derivative!(du, u, p::OscParams, t::Float64)
    N = p.N
    sx = 0.0; sy = 0.0
    @inbounds for i in 1:N
        sx += u[2i - 1]
        sy += u[2i]
    end
    xbar = sx / N
    ybar = sy / N
    @inbounds for i in 1:N
        xi  = u[2i - 1]
        yi  = u[2i]
        Ai  = envelope(p, i, t)
        ri2 = xi * xi + yi * yi
        ωi  = p.omega[i]
        du[2i - 1] = xi * (Ai - ri2) - ωi * yi + p.kappa * (xbar - xi)
        du[2i]     = yi * (Ai - ri2) + ωi * xi + p.kappa * (ybar - yi)
    end
    return du
end

# One classical fourth-order Runge–Kutta step of size `dt`, updating `u`.
function rk4step!(u, p::OscParams, t::Float64, dt::Float64, ws::Workspace)
    derivative!(ws.k1, u, p, t)
    @. ws.tmp = u + (dt / 2) * ws.k1
    derivative!(ws.k2, ws.tmp, p, t + dt / 2)
    @. ws.tmp = u + (dt / 2) * ws.k2
    derivative!(ws.k3, ws.tmp, p, t + dt / 2)
    @. ws.tmp = u + dt * ws.k3
    derivative!(ws.k4, ws.tmp, p, t + dt)
    @. u += (dt / 6) * (ws.k1 + 2 * ws.k2 + 2 * ws.k3 + ws.k4)
    return u
end

"""
    synthesize(notes, samplerate, cfg) -> Vector{Float64}

Integrate the coupled oscillator ODE driven by `notes` and return the
peak-normalised mono audio signal sampled at `samplerate` Hz.
"""
function synthesize(notes::AbstractVector{Note}, samplerate::Integer,
                    cfg::OscillatorConfig = OscillatorConfig())
    N = length(notes)
    total = isempty(notes) ? 0.0 :
            maximum(n.start + n.duration for n in notes) + cfg.release + cfg.tail
    nsamples = max(1, round(Int, total * samplerate))
    out = zeros(Float64, nsamples)
    N == 0 && return out

    p  = OscParams(notes, cfg)
    ws = Workspace(2N)
    # All oscillators start at the origin — stable under A_off < 0.
    u  = zeros(Float64, 2N)

    # Precompute per-note kick: at each note's start sample, give the oscillator
    # an initial displacement so it reaches the limit cycle within the note duration
    # rather than growing slowly from numerical noise.
    kick_sample = [max(1, round(Int, p.starts[i] * samplerate)) for i in 1:N]
    # kick to half the expected limit-cycle radius: √(vel·mu) / 2
    kick_amp    = [sqrt(max(p.vels[i] * p.mu, 0.0)) * 0.5 for i in 1:N]
    kicked      = falses(N)

    os = max(cfg.oversample, 1)
    dt = 1.0 / (samplerate * os)
    t  = 0.0
    @inbounds for k in 1:nsamples
        # Fire kick for any note whose scheduled start falls in this sample.
        for i in 1:N
            if !kicked[i] && k >= kick_sample[i]
                u[2i - 1] = kick_amp[i]
                kicked[i] = true
            end
        end
        for _ in 1:os
            rk4step!(u, p, t, dt, ws)
            t += dt
        end
        sx = 0.0
        for i in 1:N
            sx += u[2i - 1]
        end
        out[k] = sx / N
    end

    normalize_signal!(out, cfg.gain)
    return out
end

function normalize_signal!(signal, gain)
    peak = 0.0
    @inbounds for s in signal
        a = abs(s)
        a > peak && (peak = a)
    end
    if peak > 0
        scale = gain / peak
        @inbounds for i in eachindex(signal)
            signal[i] *= scale
        end
    end
    return signal
end
