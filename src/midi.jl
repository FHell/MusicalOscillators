# Minimal, dependency-free Standard MIDI File (SMF) reader and writer.
#
# Only the parts needed to drive the synthesizer are implemented: note-on /
# note-off events and tempo changes.  Both ticks-per-quarter-note (PPQ) and
# SMPTE time divisions are supported.

"""
    Note

A single played note extracted from a MIDI file.

# Fields
- `pitch::Int`     : MIDI note number (0–127, 69 == A4 == 440 Hz).
- `velocity::Int`  : note-on velocity (1–127).
- `start::Float64` : onset time in seconds.
- `duration::Float64` : duration in seconds.
- `channel::Int`   : MIDI channel (0–15).
"""
struct Note
    pitch::Int
    velocity::Int
    start::Float64
    duration::Float64
    channel::Int
end

"""
    MidiData

Parsed contents of a MIDI file: the list of [`Note`](@ref)s together with the
raw header information.
"""
struct MidiData
    notes::Vector{Note}
    format::Int
    division::Int        # raw 16-bit division field
    duration::Float64    # total length in seconds
end

Base.length(m::MidiData) = length(m.notes)

"""
    note_frequency(pitch) -> Float64

Equal-tempered frequency in Hz of a MIDI note number (A4 = 69 = 440 Hz).
"""
note_frequency(pitch::Integer) = 440.0 * 2.0^((pitch - 69) / 12)

# --- low level byte helpers ------------------------------------------------

@inline read_u16be(d, p) = (UInt16(d[p]) << 8 | UInt16(d[p + 1]), p + 2)

@inline function read_u32be(d, p)
    v = UInt32(d[p]) << 24 | UInt32(d[p + 1]) << 16 |
        UInt32(d[p + 2]) << 8 | UInt32(d[p + 3])
    return v, p + 4
end

# Variable Length Quantity (MIDI's 7-bit-per-byte integer encoding).
function read_vlq(d, p)
    val = 0
    while true
        b = d[p]; p += 1
        val = (val << 7) | (b & 0x7f)
        (b & 0x80) == 0 && break
    end
    return val, p
end

function write_vlq!(io::IO, value::Integer)
    value < 0 && throw(ArgumentError("VLQ values must be non-negative"))
    buffer = UInt8[UInt8(value & 0x7f)]
    value >>= 7
    while value > 0
        pushfirst!(buffer, UInt8((value & 0x7f) | 0x80))
        value >>= 7
    end
    write(io, buffer)
end

# --- tempo map -------------------------------------------------------------

# Convert an absolute tick to seconds given a sorted list of tempo changes
# `(tick, microseconds_per_quarter)` and the file's division field.
struct TimeMap
    ticks::Vector{Int}
    secs::Vector{Float64}
    tempos::Vector{Int}      # microseconds per quarter note
    seconds_per_tick::Float64 # used for SMPTE divisions (tempo independent)
    smpte::Bool
    division::Int
end

function TimeMap(tempos::Vector{Tuple{Int,Int}}, division::Int)
    smpte = (division & 0x8000) != 0
    if smpte
        frames = -reinterpret(Int8, UInt8((division >> 8) & 0xff))
        ticks_per_frame = division & 0xff
        spt = 1.0 / (frames * ticks_per_frame)
        return TimeMap(Int[], Float64[], Int[], spt, true, division)
    end

    sorted = sort(tempos; by = first)
    if isempty(sorted) || first(sorted[1]) > 0
        pushfirst!(sorted, (0, 500_000))   # default 120 BPM
    end

    ts = Int[t for (t, _) in sorted]
    tp = Int[v for (_, v) in sorted]
    secs = zeros(Float64, length(sorted))
    for j in 2:length(sorted)
        dtick = ts[j] - ts[j - 1]
        secs[j] = secs[j - 1] + dtick * (tp[j - 1] / 1e6) / division
    end
    return TimeMap(ts, secs, tp, 0.0, false, division)
end

function tick_to_seconds(tm::TimeMap, tick::Integer)
    tm.smpte && return tick * tm.seconds_per_tick
    # find the last tempo change at or before `tick`
    idx = searchsortedlast(tm.ticks, tick)
    idx = max(idx, 1)
    return tm.secs[idx] + (tick - tm.ticks[idx]) * (tm.tempos[idx] / 1e6) / tm.division
end

# --- reading ---------------------------------------------------------------

# An intermediate note-on/off event keyed by absolute tick.
struct RawEvent
    tick::Int
    kind::Symbol    # :on or :off
    channel::Int
    pitch::Int
    velocity::Int
end

"""
    read_midi(path) -> MidiData

Read a Standard MIDI File from `path` and return the extracted notes.
"""
read_midi(path::AbstractString) = parse_smf(read(path))

"""
    parse_smf(data::Vector{UInt8}) -> MidiData

Parse the bytes of a Standard MIDI File.
"""
function parse_smf(data::AbstractVector{UInt8})
    length(data) >= 14 || throw(ArgumentError("file too short to be a MIDI file"))
    @views data[1:4] == b"MThd" || throw(ArgumentError("missing MThd header"))

    pos = 5
    hlen, pos = read_u32be(data, pos)
    format, pos = read_u16be(data, pos)
    ntracks, pos = read_u16be(data, pos)
    division, pos = read_u16be(data, pos)
    pos = 9 + Int(hlen)   # skip any extra header bytes

    events = RawEvent[]
    tempos = Tuple{Int,Int}[]

    for _ in 1:ntracks
        pos > length(data) && break
        @views data[pos:pos + 3] == b"MTrk" || throw(ArgumentError("missing MTrk chunk"))
        pos += 4
        tlen, pos = read_u32be(data, pos)
        track_end = pos + Int(tlen)
        pos = parse_track!(events, tempos, data, pos, track_end)
        pos = track_end
    end

    division_int = Int(division)
    tm = TimeMap(tempos, division_int)
    notes = pair_notes(events, tm)
    duration = isempty(notes) ? 0.0 : maximum(n.start + n.duration for n in notes)
    return MidiData(notes, Int(format), division_int, duration)
end

function parse_track!(events, tempos, data, pos, track_end)
    abstick = 0
    status = 0x00
    while pos < track_end
        dt, pos = read_vlq(data, pos)
        abstick += dt

        b = data[pos]
        if b & 0x80 != 0          # new status byte
            status = b
            pos += 1
        end                        # otherwise running status; `pos` stays put

        if status == 0xFF          # meta event
            mtype = data[pos]; pos += 1
            mlen, pos = read_vlq(data, pos)
            if mtype == 0x51 && mlen == 3   # set tempo
                uspq = (Int(data[pos]) << 16) | (Int(data[pos + 1]) << 8) | Int(data[pos + 2])
                push!(tempos, (abstick, uspq))
            end
            pos += mlen
        elseif status == 0xF0 || status == 0xF7   # sysex
            slen, pos = read_vlq(data, pos)
            pos += slen
        else
            hi = status & 0xF0
            channel = Int(status & 0x0F)
            if hi == 0x90 || hi == 0x80 || hi == 0xA0 || hi == 0xB0 || hi == 0xE0
                d1 = Int(data[pos]); d2 = Int(data[pos + 1]); pos += 2
                if hi == 0x90 && d2 > 0
                    push!(events, RawEvent(abstick, :on, channel, d1, d2))
                elseif hi == 0x80 || (hi == 0x90 && d2 == 0)
                    push!(events, RawEvent(abstick, :off, channel, d1, 0))
                end
            elseif hi == 0xC0 || hi == 0xD0
                pos += 1
            else
                pos += 1
            end
        end
    end
    return pos
end

# Match note-on events with their note-off counterparts (FIFO per channel+pitch).
function pair_notes(events, tm::TimeMap)
    # stable sort so equal ticks keep insertion order (offs after ons handled below)
    sorted = sort(events; by = e -> e.tick, alg = MergeSort)
    pending = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()  # (chan,pitch) => [(tick,vel)]
    notes = Note[]
    last_tick = 0
    for e in sorted
        last_tick = max(last_tick, e.tick)
        key = (e.channel, e.pitch)
        if e.kind == :on
            push!(get!(pending, key, Tuple{Int,Int}[]), (e.tick, e.velocity))
        else
            q = get(pending, key, nothing)
            if q !== nothing && !isempty(q)
                start_tick, vel = popfirst!(q)
                emit_note!(notes, tm, start_tick, e.tick, vel, e.pitch, e.channel)
            end
        end
    end
    # close any notes left hanging at the final tick
    for ((channel, pitch), q) in pending
        for (start_tick, vel) in q
            emit_note!(notes, tm, start_tick, last_tick, vel, pitch, channel)
        end
    end
    sort!(notes; by = n -> n.start)
    return notes
end

function emit_note!(notes, tm, start_tick, end_tick, vel, pitch, channel)
    t0 = tick_to_seconds(tm, start_tick)
    t1 = tick_to_seconds(tm, end_tick)
    push!(notes, Note(pitch, vel, t0, max(t1 - t0, 0.0), channel))
end

# --- writing ---------------------------------------------------------------

"""
    write_midi(path, notes; tempo = 500_000, division = 480)

Write `notes` to a Type-0 Standard MIDI File.  `tempo` is in microseconds per
quarter note (500000 == 120 BPM) and `division` is ticks per quarter note.
Primarily useful for tests and for generating example input.
"""
function write_midi(path::AbstractString, notes::AbstractVector{Note};
                    tempo::Integer = 500_000, division::Integer = 480)
    sec_per_tick = (tempo / 1e6) / division
    to_tick(t) = round(Int, t / sec_per_tick)

    # build a tick-sorted list of (tick, isoff, channel, pitch, velocity)
    evs = Tuple{Int,Bool,Int,Int,Int}[]
    for n in notes
        push!(evs, (to_tick(n.start), false, n.channel, n.pitch, n.velocity))
        push!(evs, (to_tick(n.start + n.duration), true, n.channel, n.pitch, 0))
    end
    # note-offs before note-ons at the same tick keeps things clean
    sort!(evs; by = e -> (e[1], e[2] ? 0 : 1))

    track = IOBuffer()
    # tempo meta event at tick 0
    write_vlq!(track, 0)
    write(track, UInt8[0xFF, 0x51, 0x03,
                       UInt8((tempo >> 16) & 0xff),
                       UInt8((tempo >> 8) & 0xff),
                       UInt8(tempo & 0xff)])

    prev = 0
    for (tick, isoff, channel, pitch, vel) in evs
        write_vlq!(track, tick - prev)
        prev = tick
        status = UInt8((isoff ? 0x80 : 0x90) | (channel & 0x0F))
        write(track, UInt8[status, UInt8(pitch & 0x7f), UInt8(vel & 0x7f)])
    end
    # end-of-track meta event
    write_vlq!(track, 0)
    write(track, UInt8[0xFF, 0x2F, 0x00])
    track_bytes = take!(track)

    open(path, "w") do io
        write(io, b"MThd")
        write(io, hton(UInt32(6)))
        write(io, hton(UInt16(0)))                 # format 0
        write(io, hton(UInt16(1)))                 # 1 track
        write(io, hton(UInt16(division)))
        write(io, b"MTrk")
        write(io, hton(UInt32(length(track_bytes))))
        write(io, track_bytes)
    end
    return path
end
