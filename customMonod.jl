using LinearAlgebra
using Random
using Plots

# Define parameters
params = Dict(
    :muXmax => 0.5,
    :KSX => 1.0,
    :sigmaX => 0.2,
    :KσX => 0.8,

    :qYmax => 0.4,
    :KSY => 1.2,
    :sigmaY => 0.25,
    :KσY => 0.9,

    :qZmax => 0.3,
    :KSZ => 1.1,
    :sigmaZ => 0.3,
    :KσZ => 0.7,

    :YXS => 0.45,
    :YYS => 0.3,
    :mS => 0.05,
    :sigmaS => 0.2,
    :KσS => 0.8
)

# Drift function f(x, t)
function f(x, t; p=params)
    X, Y, Z, S = x

    μ = p[:muXmax] * S / (p[:KSX] + S)
    qY = p[:qYmax] * S / (p[:KSY] + S)
    qZ = p[:qZmax] * Y / (p[:KSZ] + Y)

    dX = μ * X
    dY = (qY - qZ) * X
    dZ = qZ * X
    dS = (-1/p[:YXS]) * μ * X - (1/p[:YYS]) * qY * X - p[:mS] * X

    return [dX, dY, dZ, dS]
end

# Diffusion function g(x, t): ℝ → ℝ⁴
function g(x, t; p=params)
    X, Y, Z, S = x

    G(σ, Kσ, var) = exp(-0.5 * ((Kσ - var)/σ)^2) / sqrt(2π)

    dX = (X / p[:sigmaX]) * G(p[:sigmaX], p[:KσX], S)
    dY = (X / p[:sigmaY]) * G(p[:sigmaY], p[:KσY], S)
    dZ = (X / p[:sigmaZ]) * G(p[:sigmaZ], p[:KσZ], Y)
    dS = (X / p[:sigmaS]) * G(p[:sigmaS], p[:KσS], S)

    return [dX, dY, dZ, dS]
end

# μ_X and σ_sub as functions of S
function compute_muX(S, p)
    return p[:muXmax] * S / (p[:KSX] + S)
end

function compute_sigma_sub(S, p)
    σ = p[:sigmaX]
    Kσ = p[:KσX]
    return (1 / (σ * sqrt(2π))) * exp(-0.5 * ((Kσ - S) / σ)^2)
end

# Euler-Maruyama with clipping on S equation only
function euler_maruyama_substrate_clipped(f, g, x0, tspan, dt; p=params)
    t0, tf = tspan
    N = Int(round((tf - t0) / dt))
    t = range(t0, length=N+1, step=dt)
    x = zeros(length(x0), N+1)
    x[:, 1] = x0

    for i in 1:N
        ξ = randn()
        dBt = sqrt(dt) * ξ

        S = x[4, i]  # S is the 4th state component

        μX = compute_muX(S, p)
        σsub = compute_sigma_sub(S, p)
        ξ_bound = μX * sqrt(dt) / σsub

        # Clip only the diffusion in S direction
        drift = f(x[:, i], t[i]; p=p)
        diffusion = g(x[:, i], t[i]; p=p)

        dBt_vec = diffusion .* dBt
        if abs(ξ) > ξ_bound
            ξ = sign(ξ) * ξ_bound
            dBt_vec[4] = diffusion[4] * sqrt(dt) * ξ
        end

        x[:, i+1] = x[:, i] + drift * dt + dBt_vec
    end

    return t, x
end

# Example usage
x0 = [0.1, 0.0, 0.0, 5.0]
tspan = (0.0, 10.0)
dt = 0.01

t, x = euler_maruyama_substrate_clipped(f, g, x0, tspan, dt)

plot(t, x', label=["X" "Y" "Z" "S"], xlabel="Time", ylabel="Concentration",
     title="Stochastic Bioprocess with Substrate Noise Clipping")