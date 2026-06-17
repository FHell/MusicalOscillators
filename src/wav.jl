# Minimal, dependency-free 16-bit PCM WAV reader and writer (mono).

"""
    write_wav(path, signal, samplerate)

Write `signal` (values in [-1, 1]) to a 16-bit PCM mono WAV file.
"""
function write_wav(path::AbstractString, signal::AbstractVector{<:Real},
                   samplerate::Integer)
    n = length(signal)
    data_bytes = 2 * n
    open(path, "w") do io
        # RIFF chunk descriptor
        write(io, b"RIFF")
        write(io, htol(UInt32(36 + data_bytes)))
        write(io, b"WAVE")
        # fmt subchunk
        write(io, b"fmt ")
        write(io, htol(UInt32(16)))        # subchunk size
        write(io, htol(UInt16(1)))         # PCM
        write(io, htol(UInt16(1)))         # mono
        write(io, htol(UInt32(samplerate)))
        write(io, htol(UInt32(samplerate * 2)))  # byte rate
        write(io, htol(UInt16(2)))         # block align
        write(io, htol(UInt16(16)))        # bits per sample
        # data subchunk
        write(io, b"data")
        write(io, htol(UInt32(data_bytes)))
        @inbounds for s in signal
            v = clamp(float(s), -1.0, 1.0)
            write(io, htol(round(Int16, v * 32767)))
        end
    end
    return path
end

"""
    read_wav(path) -> (signal::Vector{Float64}, samplerate::Int)

Read a 16-bit PCM mono WAV file written by [`write_wav`](@ref).
"""
function read_wav(path::AbstractString)
    data = read(path)
    @views data[1:4] == b"RIFF" || throw(ArgumentError("not a RIFF file"))
    @views data[9:12] == b"WAVE" || throw(ArgumentError("not a WAVE file"))

    samplerate = 0
    signal = Float64[]
    pos = 13
    while pos + 8 <= length(data) + 1
        id = @views data[pos:pos + 3]
        size = Int(ltoh(reinterpret(UInt32, data[pos + 4:pos + 7])[1]))
        body = pos + 8
        if id == b"fmt "
            samplerate = Int(ltoh(reinterpret(UInt32, data[body + 4:body + 7])[1]))
        elseif id == b"data"
            samples = reinterpret(Int16, data[body:body + size - 1])
            signal = Float64[ltoh(s) / 32767 for s in samples]
        end
        pos = body + size + (size % 2)   # chunks are word-aligned
    end
    return signal, samplerate
end
