module MAutils

    using Pkg
    Pkg.activate(@__DIR__)
    using DifferentialEquations, Random, Distributions, Statistics, DataFrames, CSV

    function kinetics(
        params::Dict, # Parameters for the simulation
        T::Float64, # Total time for the simulation
        u0::Dict, # Initial conditions
        dt::Float64 = 0.01) #

        # Load Experimental Data
        df = CSV.read("datasetsMA/nitrogenlim.csv", DataFrame)
        dfs = hcat(df.Xa, df.Xi, df.N, df.S, df.FG, df.MA)
        u0 = dfs[1, :]
        t = df.time
        tspan = (t[1], t[end])

        # Define ODE Model
        function f!(du, u, p, t)
            # Unpack state and parameters
            μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2, σxa, σxi, σs, σfg, σp = p
            Xact, Xinact, N, Suc, FruGlu, P = u

            # Ensure non-negative values
            ϵ = 1e-8  # Small positive value to avoid division by zero
            Xact_safe = max(Xact, ϵ)
            Xtot_safe = max(Xact + Xinact, ϵ)
            FruGlu_safe = max(FruGlu, ϵ)
            Suc_safe = max(Suc, ϵ)

            # Algebraic equations
            Xtot = Xact + Xinact
            N_int = 0.08 * N
            ratio = Xinact / Xact_safe
            expo_term = (ratio - ϕ) / χacc

            μ = μmax * FruGlu_safe / (FruGlu_safe + KFG + ϵ) * (N / (N+ KN + ϵ))
            μ2 = μ2max * FruGlu_safe / (FruGlu_safe + KFG2 + ϵ) * (1 - exp(expo_term)) * KIN / (KIN + N + ϵ)
            qsplit = qsplit_max * Suc_safe / (Suc_safe + Ksuc + ϵ)
            qp = qpmax * FruGlu_safe / (FruGlu_safe + KPFG + ϵ) *
                (KIP / (KIP + N_int / Xtot_safe + ϵ)) * KIN / (KIN + N + ϵ)

            du[1] = μ * Xact
            du[2] = μ2 * Xact
            du[3] = - (μ / YXa_N) * Xact
            du[4] = - qsplit * Xact
            du[5] = (qsplit - μ / YXa_S - μ2 / YXi_S - qp / YP_S) * Xact
            du[6] = qp * Xact
        end

        function noise!(du, u, p, t)
            Xact, Xinact, N, Suc, FruGlu, P = u
            μmax, KFG, KN, YXa_S, YXi_S, YXa_N, YP_S, ϕ, χacc, μ2max, qsplit_max, Ksuc, qpmax, KIP, KIN, KPFG, KFG2, σxa, σxi, σs, σfg, σp = p
            du[1] = σxa * FruGlu / (FruGlu + (KFG)) * Xact
            du[2] = σxi * FruGlu / (FruGlu + (KFG2)) * Xinact
            # du[3] = (σxa / YXa_N) * FruGlu / (FruGlu + (KFG)) * Xact
            # du[4] = σs * Suc / (Suc + Ksuc) * Xact
            du[5] = σfg * FruGlu / (FruGlu + Ksuc) * (Xact + Xinact)
            du[6] = σp * FruGlu / (FruGlu + (KPFG)) * Xact
        end

        # === Parameters ===
        params = [
            0.125,  # μmax
            0.147,  # KFG
            3.8e-5,  # KN
            0.531,  # YXa_S
            0.799,  # YXi_S
            9.428,  # YXa_N
            0.508,  # YP_S
            1.56,  # ϕ
            0.3,  # χacc
            0.125,  # μ2max
            1.985,  # qsplit_max
            0.00321,  # Ksuc
            0.095,  # qpmax
            1.5,  # KIP
            1.5e-3,  # KIN
            0.0175,  # KPFG
            3.277,  # KFG2
            0.05,   # σxa
            0.05,   # σxi
            0.05,   # σs
            0.05,   # σfg
            0.05    # σp
        ]

        # === Solve the ODE ===
        prob = ODEProblem(f!, u0, tspan, params)
        odesol = solve(prob, Rosenbrock23(), saveat=dt, abstol=1e-8, reltol=1e-6)

        # === Extend to SDE ===
        sdeprob = SDEProblem(f!, noise!, u0, tspan, params)
        dt = 0.1  # Time step for the SDE solver
        sdesol = solve(sdeprob, ImplicitEM(), dt=dt, saveat=dt, abstol=1e-8, reltol=1e-6)

        return sdesol, odesol

    end # end the function kinetics

    function simulate_paths(
        params::Dict, # Parameters for the simulation
        T::Float64, # Total time for the simulation
        u0::Dict, # Initial conditions
        M::Int, # Number of paths to simulate
        noise::String = "lognormal", # Type of noise to apply ("lognormal", "normal", or "monod")
        dt::Float64 = 0.01) # Default time step

        sdesolutions = Vector{Any}(undef, M)
        for i in 1:M
            sdesolutions[i] = kinetics(params, T, u0, dt)[1] # Store SDE solutions
        end
        odesolution = kinetics(params, T, u0, dt)[2] # Store ODE solution
        
        return sdesolutions, odesolution
    end # end the function simulate_paths

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
    end # end the function laguerre_design_matrix

end # end the module MAutils.jl