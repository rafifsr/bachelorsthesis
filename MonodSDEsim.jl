module MonodSDEsim
using Pkg
Pkg.activate(@__DIR__)
using DifferentialEquations

export simulate

function simulate(
    params::Dict,
    tspan::Tuple{Float64, Float64},
    u0::Dict)
    
    μ_max = params["μ_max"] 
    K_sx = params["K_sx"] 
    Y_xs = params["Y_xs"]
    Y_ys = params["Y_ys"]
    m_s = params["m_s"]
    q_max_y = params["q_max_y"]
    K_sy = params["K_sy"]
    q_max_z = params["q_max_z"]
    K_sz = params["K_sz"]
    σs = params["σs"] 
    K_σs = params["K_σs"]
    σx = params["σx"] 
    K_σx = params["K_σx"]
    σy = params["σy"]
    K_σy = params["K_σy"] 
    σz = params["σz"]
    K_σz = params["K_σz"]

    # Pack the parameters into a tuple
    p = (μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, q_max_z, K_sz, σs, K_σs, σx, K_σx, σy, K_σy, σz, K_σz)

    # Drift term
    function drift!(du, u, p, t)
        S, X, Y, Z = u

        μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, q_max_z, K_sz = p[1:9]
       
        μ_x = μ_max * (S / (K_sx + S))
        q_py = q_max_y * (S / (K_sy + S))
        q_pz = q_max_z * (Y / (K_sz + Y))
        
        # BioVT
        dS = - (1 / Y_xs) * μ_x * X - (1 / Y_ys) * q_py * X - m_s * X
        dX = μ_x * X
        dY = q_py * X - q_pz * X
        dZ = q_pz * X

        du[1] = dS
        du[2] = dX
        du[3] = dY
        du[4] = dZ
    end

    # Diffusion term
    function diffusion!(du, u, p, t)
        S, X, Y, Z = u
        σs, K_σs, σx, K_σx, σy, K_σy, σz, K_σz = p[10:17]
        du[1] = σs * (S / (K_σs + S)) * X
        du[2] = σx * (S / (K_σx + S)) * X
        du[3] = σy * (S / (K_σy + S)) * X
        du[4] = σz * (Y / (K_σz + Y)) * X
    end

    # Initial conditions
    S0, X0, Y0, Z0 = u0["S0"], u0["X0"], u0["Y0"], u0["Z0"]
    u_0 = [S0, X0, Y0, Z0]  # Initial concentrations of S, X, Y, and Z

    # Problem definition
    prob = SDEProblem(drift!, diffusion!, u_0, tspan, p)
    sol = solve(prob, EM(), dt=0.01, saveat=0.1)
    return sol

end # function simulate

end # module MonodSDEsim