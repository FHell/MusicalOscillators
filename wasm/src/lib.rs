use wasm_bindgen::prelude::*;

/// Convert MIDI pitch to frequency in Hz.
/// note_frequency(69) = 440.0 Hz (A4)
#[wasm_bindgen]
pub fn note_frequency(pitch: u8) -> f64 {
    440.0 * 2f64.powf((pitch as f64 - 69.0) / 12.0)
}

/// Signed Stuart-Landau envelope A_i(t) for note i.
/// > 0 during note (stable limit cycle at r = sqrt(A))
/// < 0 between notes (oscillator decays to zero)
fn envelope(
    t: f64,
    start: f64,
    dur: f64,
    vel: f64,
    mu: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
) -> f64 {
    let end = start + dur;
    let a_on = vel * mu;
    let a_off = -damp;

    if t < start {
        a_off
    } else if t < start + attack {
        let alpha = (t - start) / attack;
        a_off + alpha * (a_on - a_off)
    } else if t <= end {
        a_on
    } else {
        // level at note-off (may not have reached a_on if attack > dur)
        let held = if dur < attack {
            a_off + (dur / attack) * (a_on - a_off)
        } else {
            a_on
        };
        let rt = t - end;
        if rt < release_time {
            held + (rt / release_time) * (a_off - held)
        } else {
            a_off
        }
    }
}

/// Compute derivative of the full coupled Stuart-Landau system (scalar coupling).
fn derivative(
    du: &mut Vec<f64>,
    u: &[f64],
    n: usize,
    omega: &[f64],
    starts: &[f64],
    durs: &[f64],
    vels: &[f64],
    mu: f64,
    kappa: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    t: f64,
) {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..n {
        sx += u[2 * i];
        sy += u[2 * i + 1];
    }
    let xbar = sx / n as f64;
    let ybar = sy / n as f64;

    for i in 0..n {
        let xi = u[2 * i];
        let yi = u[2 * i + 1];
        let ai = envelope(t, starts[i], durs[i], vels[i], mu, damp, attack, release_time);
        let ri2 = xi * xi + yi * yi;
        let wi = omega[i];
        du[2 * i] = xi * (ai - ri2) - wi * yi + kappa * (xbar - xi);
        du[2 * i + 1] = yi * (ai - ri2) + wi * xi + kappa * (ybar - yi);
    }
}

/// Compute derivative with matrix coupling (N×N matrix, row-major).
/// Term for oscillator i: coupling_scale * sum_j K[i,j] * (x_j - x_i)
fn derivative_matrix(
    du: &mut Vec<f64>,
    u: &[f64],
    n: usize,
    omega: &[f64],
    starts: &[f64],
    durs: &[f64],
    vels: &[f64],
    mu: f64,
    coupling_matrix: &[f64],
    matrix_size: usize,
    coupling_scale: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    t: f64,
) {
    for i in 0..n {
        let xi = u[2 * i];
        let yi = u[2 * i + 1];
        let ai = envelope(t, starts[i], durs[i], vels[i], mu, damp, attack, release_time);
        let ri2 = xi * xi + yi * yi;
        let wi = omega[i];

        // Coupling: sum over j of K[i,j] * (x_j - x_i)
        // Use matrix index clamped to matrix_size
        let mi = i.min(matrix_size - 1);
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        for j in 0..n {
            let mj = j.min(matrix_size - 1);
            let k_ij = coupling_matrix[mi * matrix_size + mj];
            cx += k_ij * (u[2 * j] - xi);
            cy += k_ij * (u[2 * j + 1] - yi);
        }

        du[2 * i] = xi * (ai - ri2) - wi * yi + coupling_scale * cx;
        du[2 * i + 1] = yi * (ai - ri2) + wi * xi + coupling_scale * cy;
    }
}

fn rk4_step_scalar(
    u: &mut Vec<f64>,
    tmp: &mut Vec<f64>,
    k1: &mut Vec<f64>,
    k2: &mut Vec<f64>,
    k3: &mut Vec<f64>,
    k4: &mut Vec<f64>,
    n: usize,
    omega: &[f64],
    starts: &[f64],
    durs: &[f64],
    vels: &[f64],
    mu: f64,
    kappa: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    t: f64,
    dt: f64,
) {
    let dim = 2 * n;
    derivative(k1, u, n, omega, starts, durs, vels, mu, kappa, damp, attack, release_time, t);
    for i in 0..dim { tmp[i] = u[i] + (dt / 2.0) * k1[i]; }
    derivative(k2, tmp, n, omega, starts, durs, vels, mu, kappa, damp, attack, release_time, t + dt / 2.0);
    for i in 0..dim { tmp[i] = u[i] + (dt / 2.0) * k2[i]; }
    derivative(k3, tmp, n, omega, starts, durs, vels, mu, kappa, damp, attack, release_time, t + dt / 2.0);
    for i in 0..dim { tmp[i] = u[i] + dt * k3[i]; }
    derivative(k4, tmp, n, omega, starts, durs, vels, mu, kappa, damp, attack, release_time, t + dt);
    for i in 0..dim {
        u[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

fn rk4_step_matrix(
    u: &mut Vec<f64>,
    tmp: &mut Vec<f64>,
    k1: &mut Vec<f64>,
    k2: &mut Vec<f64>,
    k3: &mut Vec<f64>,
    k4: &mut Vec<f64>,
    n: usize,
    omega: &[f64],
    starts: &[f64],
    durs: &[f64],
    vels: &[f64],
    mu: f64,
    coupling_matrix: &[f64],
    matrix_size: usize,
    coupling_scale: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    t: f64,
    dt: f64,
) {
    let dim = 2 * n;
    derivative_matrix(k1, u, n, omega, starts, durs, vels, mu, coupling_matrix, matrix_size, coupling_scale, damp, attack, release_time, t);
    for i in 0..dim { tmp[i] = u[i] + (dt / 2.0) * k1[i]; }
    derivative_matrix(k2, tmp, n, omega, starts, durs, vels, mu, coupling_matrix, matrix_size, coupling_scale, damp, attack, release_time, t + dt / 2.0);
    for i in 0..dim { tmp[i] = u[i] + (dt / 2.0) * k2[i]; }
    derivative_matrix(k3, tmp, n, omega, starts, durs, vels, mu, coupling_matrix, matrix_size, coupling_scale, damp, attack, release_time, t + dt / 2.0);
    for i in 0..dim { tmp[i] = u[i] + dt * k3[i]; }
    derivative_matrix(k4, tmp, n, omega, starts, durs, vels, mu, coupling_matrix, matrix_size, coupling_scale, damp, attack, release_time, t + dt);
    for i in 0..dim {
        u[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

fn normalize_signal(signal: &mut [f32], gain: f64) {
    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let scale = (gain as f32) / peak;
        for s in signal.iter_mut() {
            *s *= scale;
        }
    }
}

fn core_synthesize(
    pitches: &[u8],
    velocities: &[u8],
    starts: &[f64],
    durations: &[f64],
    samplerate: u32,
    mu: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    tail: f64,
    gain: f64,
    step_fn: &mut dyn FnMut(&mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>, usize, &[f64], &[f64], &[f64], &[f64], f64, f64, f64, f64, f64),
) -> Vec<f32> {
    let n = pitches.len();
    if n == 0 {
        return Vec::new();
    }

    let omega: Vec<f64> = pitches.iter().map(|&p| 2.0 * std::f64::consts::PI * note_frequency(p)).collect();
    let vels: Vec<f64> = velocities.iter().map(|&v| (v as f64 / 127.0).clamp(0.0, 1.0)).collect();

    let total = starts.iter().zip(durations.iter())
        .map(|(&s, &d)| s + d)
        .fold(0.0f64, f64::max)
        + release_time + tail;

    let nsamples = (total * samplerate as f64).round() as usize;
    let nsamples = nsamples.max(1);

    let mut out = vec![0.0f32; nsamples];
    let dim = 2 * n;
    let mut u = vec![0.0f64; dim];
    let mut tmp = vec![0.0f64; dim];
    let mut k1 = vec![0.0f64; dim];
    let mut k2 = vec![0.0f64; dim];
    let mut k3 = vec![0.0f64; dim];
    let mut k4 = vec![0.0f64; dim];

    let kick_samples: Vec<usize> = starts.iter().map(|&s| ((s * samplerate as f64).round() as usize).max(0)).collect();
    let kick_amps: Vec<f64> = vels.iter().map(|&v| (v * mu).max(0.0).sqrt() * 0.5).collect();
    let mut kicked = vec![false; n];

    let os = 2usize;
    let dt = 1.0 / (samplerate as f64 * os as f64);
    let mut t = 0.0f64;

    for k in 0..nsamples {
        // Apply kicks
        for i in 0..n {
            if !kicked[i] && k >= kick_samples[i] {
                u[2 * i] = kick_amps[i];
                kicked[i] = true;
            }
        }
        // RK4 substeps
        for _ in 0..os {
            step_fn(&mut u, &mut tmp, &mut k1, &mut k2, &mut k3, &mut k4,
                    n, &omega, starts, durations, &vels, mu, damp, attack, release_time, t, dt);
            t += dt;
        }
        // Output: mean of x_i
        let sx: f64 = (0..n).map(|i| u[2 * i]).sum();
        out[k] = (sx / n as f64) as f32;
    }

    normalize_signal(&mut out, gain);
    out
}

/// Synthesize audio using the Stuart-Landau oscillator bank with scalar mean-field coupling.
#[wasm_bindgen]
pub fn synthesize(
    pitches: &[u8],
    velocities: &[u8],
    starts: &[f64],
    durations: &[f64],
    samplerate: u32,
    mu: f64,
    coupling: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    tail: f64,
    gain: f64,
) -> Vec<f32> {
    let n = pitches.len();
    if n == 0 {
        return Vec::new();
    }

    let omega: Vec<f64> = pitches.iter().map(|&p| 2.0 * std::f64::consts::PI * note_frequency(p)).collect();
    let vels: Vec<f64> = velocities.iter().map(|&v| (v as f64 / 127.0).clamp(0.0, 1.0)).collect();

    let total = starts.iter().zip(durations.iter())
        .map(|(&s, &d)| s + d)
        .fold(0.0f64, f64::max)
        + release_time + tail;

    let nsamples = (total * samplerate as f64).round() as usize;
    let nsamples = nsamples.max(1);

    let mut out = vec![0.0f32; nsamples];
    let dim = 2 * n;
    let mut u = vec![0.0f64; dim];
    let mut tmp = vec![0.0f64; dim];
    let mut k1 = vec![0.0f64; dim];
    let mut k2 = vec![0.0f64; dim];
    let mut k3 = vec![0.0f64; dim];
    let mut k4 = vec![0.0f64; dim];

    let kick_samples: Vec<usize> = starts.iter().map(|&s| ((s * samplerate as f64).round() as usize).max(0)).collect();
    let kick_amps: Vec<f64> = vels.iter().map(|&v| (v * mu).max(0.0).sqrt() * 0.5).collect();
    let mut kicked = vec![false; n];

    let os = 2usize;
    let dt = 1.0 / (samplerate as f64 * os as f64);
    let mut t = 0.0f64;

    for k in 0..nsamples {
        for i in 0..n {
            if !kicked[i] && k >= kick_samples[i] {
                u[2 * i] = kick_amps[i];
                kicked[i] = true;
            }
        }
        for _ in 0..os {
            rk4_step_scalar(
                &mut u, &mut tmp, &mut k1, &mut k2, &mut k3, &mut k4,
                n, &omega, starts, durations, &vels,
                mu, coupling, damp, attack, release_time, t, dt,
            );
            t += dt;
        }
        let sx: f64 = (0..n).map(|i| u[2 * i]).sum();
        out[k] = (sx / n as f64) as f32;
    }

    normalize_signal(&mut out, gain);
    out
}

/// Synthesize audio using the Stuart-Landau oscillator bank with a full coupling matrix.
/// `coupling_matrix_flat` is a row-major N×N (or 12×12) matrix of coupling weights.
/// The derivative coupling term for oscillator i is:
///   coupling_scale * sum_j K[i,j] * (x_j - x_i)
#[wasm_bindgen]
pub fn synthesize_with_matrix(
    pitches: &[u8],
    velocities: &[u8],
    starts: &[f64],
    durations: &[f64],
    samplerate: u32,
    mu: f64,
    coupling_matrix_flat: &[f64],
    coupling_scale: f64,
    damp: f64,
    attack: f64,
    release_time: f64,
    tail: f64,
    gain: f64,
) -> Vec<f32> {
    let n = pitches.len();
    if n == 0 {
        return Vec::new();
    }

    // Determine matrix size from the flat array (assumed square)
    let matrix_size = (coupling_matrix_flat.len() as f64).sqrt().round() as usize;
    let matrix_size = matrix_size.max(1);

    let omega: Vec<f64> = pitches.iter().map(|&p| 2.0 * std::f64::consts::PI * note_frequency(p)).collect();
    let vels: Vec<f64> = velocities.iter().map(|&v| (v as f64 / 127.0).clamp(0.0, 1.0)).collect();

    let total = starts.iter().zip(durations.iter())
        .map(|(&s, &d)| s + d)
        .fold(0.0f64, f64::max)
        + release_time + tail;

    let nsamples = (total * samplerate as f64).round() as usize;
    let nsamples = nsamples.max(1);

    let mut out = vec![0.0f32; nsamples];
    let dim = 2 * n;
    let mut u = vec![0.0f64; dim];
    let mut tmp = vec![0.0f64; dim];
    let mut k1 = vec![0.0f64; dim];
    let mut k2 = vec![0.0f64; dim];
    let mut k3 = vec![0.0f64; dim];
    let mut k4 = vec![0.0f64; dim];

    let kick_samples: Vec<usize> = starts.iter().map(|&s| ((s * samplerate as f64).round() as usize).max(0)).collect();
    let kick_amps: Vec<f64> = vels.iter().map(|&v| (v * mu).max(0.0).sqrt() * 0.5).collect();
    let mut kicked = vec![false; n];

    let os = 2usize;
    let dt = 1.0 / (samplerate as f64 * os as f64);
    let mut t = 0.0f64;

    for k in 0..nsamples {
        for i in 0..n {
            if !kicked[i] && k >= kick_samples[i] {
                u[2 * i] = kick_amps[i];
                kicked[i] = true;
            }
        }
        for _ in 0..os {
            rk4_step_matrix(
                &mut u, &mut tmp, &mut k1, &mut k2, &mut k3, &mut k4,
                n, &omega, starts, durations, &vels,
                mu, coupling_matrix_flat, matrix_size, coupling_scale,
                damp, attack, release_time, t, dt,
            );
            t += dt;
        }
        let sx: f64 = (0..n).map(|i| u[2 * i]).sum();
        out[k] = (sx / n as f64) as f32;
    }

    normalize_signal(&mut out, gain);
    out
}
