# The synthesis core.
#
# Each MIDI note is given **two** state dimensions `(x_i, y_i)` and modelled as
# a Van-der-Pol-style self-sustaining oscillator whose natural frequency is the
# pitch of the note.  Stacking every note gives one large, coupled ODE
#
#     dx_i/dt = y_i
#     dy_i/dt = μ (a_i(t)² − x_i²) y_i − ω_i² x_i + κ (x̄ − x_i)
#
# where
#   * ω_i      is the angular frequency of note i,
#   * a_i(t)   is a velocity-scaled attack/release envelope (the MIDI signal
#              "manipulating" the oscillator — non-zero only while the note
#              sounds),
#   * x̄        is the mean of all positions (mean-field coupling that ties the
#              dimensions together into a single higher-dimensional system),
#   * μ, κ     control harmonic richness and coupling strength.
#
# The audio output is the **average** of every note's position, `mean_i x_i`.

"""
    OscillatorConfig(; kwargs...)

Parameters controlling the oscillator bank.

- `mu`          : Van der Pol nonlinearity; larger ⇒ richer harmonics (default 3.0).
- `coupling`    : mean-field coupling strength κ between oscillators (default 0.02).
- `attack`      : note attack time in seconds (default 0.01).
- `release`     : note release time in seconds (default 0.15).
- `oversample`  : ODE substeps per audio sample for stability (default 2).
- `tail`        : extra seconds rendered after the last note ends (default 0.3).
- `gain`        : peak normalisation target in [0, 1] (default 0.9).
"""
Base.@kwdef struct OscillatorConfig
    mu::Float64 = 3.0
    coupling::Float64 = 0.02
    attack::Float64 = 0.01
    release::Float64 = 0.15
    oversample::Int = 2
    tail::Float64 = 0.3
    gain::Float64 = 0.9
end

# Flattened, integration-ready parameters.
struct OscParams
    N::Int
    omega2::Vector{Float64}
    starts::Vector{Float64}
    durs::Vector{Float64}
    vels::Vector{Float64}    # normalised to [0, 1]
    mu::Float64
    kappa::Float64
    attack::Float64
    release::Float64
end

function OscParams(notes::AbstractVector{Note}, cfg::OscillatorConfig)
    N = length(notes)
    omega2 = Vector{Float64}(undef, N)
    starts = Vector{Float64}(undef, N)
    durs = Vector{Float64}(undef, N)
    vels = Vector{Float64}(undef, N)
    for (i, n) in enumerate(notes)
        ω = 2π * note_frequency(n.pitch)
        omega2[i] = ω * ω
        starts[i] = n.start
        durs[i] = n.duration
        vels[i] = clamp(n.velocity / 127, 0.0, 1.0)
    end
    return OscParams(N, omega2, starts, durs, vels, cfg.mu, cfg.coupling,
                     cfg.attack, cfg.release)
end

# Velocity-scaled attack/sustain/release envelope for note `i` at time `t`.
@inline function envelope(p::OscParams, i::Int, t::Float64)
    s = p.starts[i]
    t < s && return 0.0
    d = p.durs[i]
    v = p.vels[i]
    a = p.attack
    e = s + d
    if t <= e
        return t < s + a ? v * (t - s) / a : v
    end
    held = d < a ? v * d / a : v        # level reached at note-off
    rt = t - e
    rt < p.release ? held * (1 - rt / p.release) : 0.0
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

# In-place derivative of the full coupled system.
function derivative!(du, u, p::OscParams, t::Float64)
    N = p.N
    sx = 0.0
    @inbounds for i in 1:N
        sx += u[2i - 1]
    end
    xbar = sx / N
    @inbounds for i in 1:N
        xi = u[2i - 1]
        yi = u[2i]
        a = envelope(p, i, t)
        du[2i - 1] = yi
        du[2i] = p.mu * (a * a - xi * xi) * yi - p.omega2[i] * xi + p.kappa * (xbar - xi)
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

    p = OscParams(notes, cfg)
    ws = Workspace(2N)
    u = zeros(Float64, 2N)
    @inbounds for i in 1:N
        u[2i - 1] = 1e-3        # tiny seed so the limit cycles can start
    end

    os = max(cfg.oversample, 1)
    dt = 1.0 / (samplerate * os)
    t = 0.0
    @inbounds for k in 1:nsamples
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
