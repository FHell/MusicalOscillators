# Piano UI for MusicalOscillators.
#
#   julia --project=ui ui/app.jl
#
# Keyboard: one octave C4-B4 (MIDI 60-71).
# Left panel  : piano keyboard — click white/black keys to toggle notes.
# Right panel : 12×12 coupling matrix heatmap — scroll wheel on a cell adjusts
#               K[row,col] by ±0.05.  Bottom controls: coupling scale slider,
#               "Shuffle matrix" button, "All notes off" button.

using GLMakie
using MusicalOscillators
using PortAudio

# ── Constants ────────────────────────────────────────────────────────────────

const PITCHES     = collect(60:71)           # C4 – B4
const N_OSC       = length(PITCHES)
const SAMPLERATE  = 44100
const BLOCK_SIZE  = 512
const NOTE_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Which of the 12 semitones are black keys?
const IS_BLACK = [false,true,false,true,false,false,true,false,true,false,true,false]

# ── Build UI ─────────────────────────────────────────────────────────────────

function build_ui()
    synth = RealtimeSynth(PITCHES; samplerate = SAMPLERATE, coupling_scale = 0.02)
    note_on = Observable(fill(false, N_OSC))   # which keys are pressed

    fig = Figure(resolution = (1200, 680), backgroundcolor = :grey15)

    # ── Left half: piano keyboard ─────────────────────────────────────────────
    ax_keys = Axis(fig[1, 1];
                   title = "Piano  (click to play)",
                   aspect = DataAspect(),
                   limits = (-0.5, 7.5, -0.5, 4.5),
                   xgridvisible = false, ygridvisible = false,
                   xticksvisible = false, yticksvisible = false,
                   xticklabelsvisible = false, yticklabelsvisible = false,
                   leftspinevisible = false, rightspinevisible = false,
                   topspinevisible = false, bottomspinevisible = false,
                   backgroundcolor = :grey20)

    # White key layout (7 keys = C D E F G A B → indices 1,3,5,6,8,10,12)
    white_order = [1,3,5,6,8,10,12]   # semitone index (1-based) among 12
    white_x = collect(0:6)             # x centres of white keys

    # Draw keys; keep poly handles for recolouring
    white_polys = Poly[]
    black_polys = Poly[]
    white_idx   = Int[]   # oscillator index for each white key
    black_idx   = Int[]

    for (col, semi0) in enumerate(white_order)
        i = semi0   # 1-based semitone
        pts = Point2f[(col-1, 0), (col, 0), (col, 4), (col-1, 4)]
        c = Observable(RGBAf(1, 1, 1, 1))
        p = poly!(ax_keys, pts; color = c, strokecolor = :black, strokewidth = 2)
        push!(white_polys, p)
        push!(white_idx, i)
        text!(ax_keys, col - 0.5, 0.3; text = NOTE_NAMES[i],
              fontsize = 11, align = (:center, :bottom), color = :black)
    end

    # Black keys
    black_xmap = Dict(2 => 0.65, 4 => 1.65, 7 => 3.65, 9 => 4.65, 11 => 5.65)
    for (semi0, bx) in sort(collect(black_xmap); by = first)
        i = semi0
        pts = Point2f[(bx, 2.2), (bx+0.7, 2.2), (bx+0.7, 4), (bx, 4)]
        c = Observable(RGBAf(0.1, 0.1, 0.1, 1))
        p = poly!(ax_keys, pts; color = c, strokecolor = :black, strokewidth = 1)
        push!(black_polys, p)
        push!(black_idx, i)
    end

    # ── Right half: coupling matrix ───────────────────────────────────────────
    coupling_scale = Observable(0.02)
    K_obs = Observable(copy(synth.K))

    ax_mat = Axis(fig[1, 2];
                  title = "Coupling matrix K  (scroll to edit)",
                  xlabel = "To oscillator →",
                  ylabel = "From oscillator →",
                  xticks = (0.5:11.5, NOTE_NAMES),
                  yticks = (0.5:11.5, NOTE_NAMES),
                  aspect = DataAspect(),
                  xticklabelsize = 9, yticklabelsize = 9)

    hm = heatmap!(ax_mat, K_obs; colorrange = (0, 1), colormap = :inferno)
    Colorbar(fig[1, 3], hm; label = "K value")

    # ── Bottom controls ───────────────────────────────────────────────────────
    ctrl = GridLayout(fig[2, 1:2])

    Label(ctrl[1, 1], "Coupling scale:"; tellwidth = false)
    sl = Slider(ctrl[1, 2]; range = 0.0:0.005:0.2, startvalue = 0.02,
                width = 260)
    coupling_scale_label = Label(ctrl[1, 3], @lift(string(round($(sl.value); digits=3)));
                                  tellwidth = false)

    btn_shuffle = Button(ctrl[1, 4]; label = "Shuffle matrix", width = 130)
    btn_alloff  = Button(ctrl[1, 5]; label = "All notes off",  width = 130)
    btn_clear   = Button(ctrl[1, 6]; label = "Clear matrix",   width = 130)

    rowsize!(fig.layout, 2, Fixed(60))
    colsize!(fig.layout, 1, Relative(0.45))
    colsize!(fig.layout, 2, Relative(0.45))

    # ── Key press helpers ─────────────────────────────────────────────────────
    function set_note!(osc_i, pressed::Bool)
        v = copy(note_on[])
        v[osc_i] = pressed
        note_on[] = v
        if pressed
            activate_note!(synth, osc_i, 1.0)
        else
            deactivate_note!(synth, osc_i)
        end
    end

    # Toggle helpers for both white and black polys
    function handle_key_click(osc_i, poly_handle, default_color)
        register_interaction!(ax_keys, Symbol("key_$osc_i")) do event, _
            if event isa MouseEvent && event.type == MouseEventTypes.leftclick
                pt = event.data
                # check if click is within this key's polygon — simpler: handled
                # via individual poly mousedown below
            end
        end
        on(events(ax_keys).mousebutton) do btn
            false  # placeholder; real dispatch below
        end
    end

    # Simpler approach: respond to any mousedown on the axis and hit-test ourselves.
    white_bounds = [(i-1, i, 0, 4)  for i in 1:7]  # (xmin,xmax,ymin,ymax) in data coords
    black_bounds_list = [(bx, bx+0.7, 2.2, 4.0) for (_, bx) in
                         sort(collect(black_xmap); by=first)]

    on(events(fig).mousebutton) do btn
        btn.action == Mouse.press || return
        mp = mouseposition(ax_keys.scene)
        x, y = mp

        # Hit-test black keys first (they sit on top)
        for (k, (x0, x1, y0, y1)) in enumerate(black_bounds_list)
            if x0 ≤ x ≤ x1 && y0 ≤ y ≤ y1
                i = black_idx[k]
                set_note!(i, !note_on[][i])
                col = note_on[][i] ? RGBAf(0.4, 0.4, 0.9, 1) : RGBAf(0.1, 0.1, 0.1, 1)
                black_polys[k].color[] = col
                return
            end
        end
        # White keys
        for (k, (x0, x1, y0, y1)) in enumerate(white_bounds)
            if x0 ≤ x ≤ x1 && y0 ≤ y ≤ y1
                i = white_idx[k]
                set_note!(i, !note_on[][i])
                col = note_on[][i] ? RGBAf(0.5, 0.6, 1.0, 1) : RGBAf(1, 1, 1, 1)
                white_polys[k].color[] = col
                return
            end
        end
    end

    # ── Matrix scroll interaction ─────────────────────────────────────────────
    on(events(fig).scroll) do delta
        mp = mouseposition(ax_mat.scene)
        x, y = mp
        col = floor(Int, x) + 1   # 1-based column = "to" oscillator
        row = floor(Int, y) + 1   # 1-based row    = "from" oscillator
        (1 ≤ row ≤ N_OSC && 1 ≤ col ≤ N_OSC) || return
        set_coupling!(synth, row, col, synth.K[row, col] + 0.05 * sign(delta[2]))
        K_obs[] = copy(synth.K)
    end

    # ── Button actions ────────────────────────────────────────────────────────
    on(btn_shuffle.clicks) do _
        shuffle_coupling!(synth)
        K_obs[] = copy(synth.K)
    end

    on(btn_clear.clicks) do _
        lock(synth.lock) do
            fill!(synth.K, 0.0)
        end
        K_obs[] = copy(synth.K)
    end

    on(btn_alloff.clicks) do _
        for i in 1:N_OSC
            deactivate_note!(synth, i)
            note_on[] = fill(false, N_OSC)
        end
        for (k, p) in enumerate(white_polys)
            p.color[] = RGBAf(1, 1, 1, 1)
        end
        for (k, p) in enumerate(black_polys)
            p.color[] = RGBAf(0.1, 0.1, 0.1, 1)
        end
    end

    on(sl.value) do v
        lock(synth.lock) do
            synth.coupling_scale = v
        end
    end

    # ── Audio loop ────────────────────────────────────────────────────────────
    buf = zeros(Float64, BLOCK_SIZE)
    buf16 = zeros(Int16, BLOCK_SIZE)

    audio_task = @async begin
        PortAudio.PortAudioStream(0, 1; samplerate = SAMPLERATE,
                                  buffersize = BLOCK_SIZE) do stream
            while isopen(stream)
                generate_samples!(synth, buf)
                # soft-clip to avoid crackle on loud chords
                @inbounds for i in eachindex(buf)
                    buf16[i] = round(Int16, clamp(buf[i], -1.0, 1.0) * 32767)
                end
                write(stream, reshape(buf16, 1, :))
            end
        end
    end

    return fig, audio_task
end

fig, audio_task = build_ui()
display(fig)
wait(fig.scene)
