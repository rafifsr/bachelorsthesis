module MonodSDEsim
using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations

export simulate

function simulate(
    params::Dict,
    tspan::Tuple{Float64, Float64},
    u0::Dict)
    
    # Unpack parameters
    μ_max, K_s, Y_sx, q_max, K_sp, Y_sp, k_d, k_z, σ1, σ2, σ3 = params["μ_max"], params["K_s"], params["Y_sx"], params["q_max"], params["K_sp"], params["Y_sp"], params["k_d"], params["k_z"], params["σ1"], params["σ2"], params["σ3"]
    p = (μ_max, K_s, Y_sx, q_max, K_sp, Y_sp, k_d, k_z, σ1, σ2, σ3)

    # Drift term
    function drift!(du, u, p, t)
        S, X, P, Z = u
       
        µ = μ_max * S / (K_s + S)  # Monod growth rate
        q = q_max * S / (K_sp + S)  # BioVT product formation rate
        
        # BioVT
        dS = - (1 / Y_sx) * μ * X - (1 / Y_sp) * q * X
        dX = μ * X - k_z * P * X - k_d * X
        dP = q * X - k_z * P * X
        dZ = k_z * P * X

        du[1] = dS
        du[2] = dX
        du[3] = dP
        du[4] = dZ
    end

    # Diffusion term
    function diffusion!(du, u, p, t)
        S, X, P, Z = u
        du[1] = -(σ1 + σ2) * X * S
        du[2] = σ1 * X * S
        du[3] = σ2 * X * S
        du[4] = σ3 * X * P
    end

    # Initial conditions
    S0, X0, P0, Z0 = u0["S0"], u0["X0"], u0["P0"], u0["Z0"]
    u_0 = [S0, X0, P0, Z0]  # Initial concentrations of S, X, P, and Z

    # Problem definition
    prob = SDEProblem(drift!, diffusion!, u_0, tspan, p)
    sol = solve(prob, EM(), dt=0.01, saveat=0.1)
    return sol

end # function simulate

end # module MonodSDEsim