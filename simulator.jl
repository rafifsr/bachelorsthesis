module simulator
    using Pkg
    Pkg.activate(@__DIR__)
    using DifferentialEquations, Random, Distributions, Statistics

    export monod, simulate_paths, laguerre_design_matrix

    function monod(
        params::Dict,
        tspan::Tuple{Float64, Float64},
        u0::Dict,
        dt::Float64 = 0.01)
        
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
        tsteps = tspan[1]:dt:tspan[2]  # Time steps for the simulation

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
            μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, q_max_z, K_sz, σs, K_σs, σx, K_σx, σy, K_σy, σz, K_σz = p
            µ_s = μ_max/Y_xs + q_max_y/Y_ys
            du[1] = σs * pdf(LogNormal(log(K_σs) + µ_s^2, µ_s), S) * X
            du[2] = σx * pdf(LogNormal(log(K_σx) + μ_max^2, μ_max), S) * X
            du[3] = σy * pdf(LogNormal(log(K_σy) + q_max_y^2, q_max_y), S) * X
            du[4] = σz * pdf(LogNormal(log(K_σz) + q_max_z^2, q_max_z), Y) * X
        end

        # Initial conditions
        S0, X0, Y0, Z0 = u0["S0"], u0["X0"], u0["Y0"], u0["Z0"]
        u_0 = [S0, X0, Y0, Z0]  # Initial concentrations of S, X, Y, and Z

        # Problem definition
        prob = SDEProblem(drift!, diffusion!, u_0, tspan, p)
        sol = solve(prob, EM(), dt=dt, saveat=tsteps)
        return sol
    end # function monod

    # Run M simulations and collect results
    function simulate_paths(
        params::Dict, 
        tspan::Tuple{Float64, Float64},
        u0::Dict, 
        M::Int,
        dt::Float64 = 0.01)

        results = Vector{Any}(undef, M)
        for i in 1:M
            results[i] = monod(params, tspan, u0, dt)
        end
        return results
    end # function simulate_paths

    # Laguerre basis functions
    function laguerre_design_matrix(
        y::Vector{Float64}, 
        d::Int)

        Φ = zeros(length(y), d + 1)
        for i in 1:length(y)
            Φ[i,1] = 1.0
            if d >= 1
                Φ[i,2] = 1 - y[i]
            end
            if d >= 2
                Φ[i,3] = 1 - 2*y[i] + 0.5*y[i]^2
            end
            if d >= 3
                Φ[i,4] = 1 - 3*y[i] + 1.5*y[i]^2 - (1/6)*y[i]^3
            end
        end
        return Φ
    end # function laguerre_design_matrix

end # module simulator