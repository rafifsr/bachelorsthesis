module SDEutils

    using Pkg
    Pkg.activate(@__DIR__)
    using DifferentialEquations, Random, Distributions, Statistics

    export kinetics, simulate_paths, laguerre_design_matrix

    function kinetics(
        params::Dict, # Parameters for the kinetics
        T::Float64, # Total time for the simulation
        u0::Dict, # Initial conditions
        noise::String, # Type of noise to apply ("lognormal", "normal", or "monod")
        dt::Float64 = 0.01) # Default time step
        
        # Biomass kinetics
        μ_max = params["μ_max"] 
        K_sx = params["K_sx"] 
        
        # Substrate kinetics
        Y_xs = params["Y_xs"]
        Y_ys = params["Y_ys"]
        m_s = params["m_s"]
        
        # Product kinetics
        q_max_y = params["q_max_y"]
        K_sy = params["K_sy"]

        # By-product kinetics
        q_max_z = params["q_max_z"]
        K_sz = params["K_sz"]

        # Noise scaling parameters
        σs = params["σs"]
        σx = params["σx"]
        σy = params["σy"]
        σz = params["σz"]

        # Substrate noise parameters
        K_ss = K_sx/Y_xs + K_sy/Y_ys
        µ_s = μ_max/Y_xs + q_max_y/Y_ys

        # Pack the parameters into a tuple
        p = (μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, q_max_z, K_sz, σs, σx, σy, σz, K_ss, µ_s)

        # Initial conditions
        S0, X0, Y0, Z0 = u0["S0"], u0["X0"], u0["Y0"], u0["Z0"]
        u_0 = [S0, X0, Y0, Z0]

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
            dY = q_py * X - q_pz * Y
            dZ = q_pz * Y
            
            du[1] = dS
            du[2] = dX
            du[3] = dY
            du[4] = dZ
        end

        # Diffusion term (lognormal)
        function lognormal!(du, u, p, t)
            S, X, Y, Z = u
            μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, q_max_z, K_sz, σs, σx, σy, σz, K_ss, µ_s = p
            du[1] = σs * pdf(LogNormal(log(K_ss) + µ_s^2, µ_s), S) * X
            du[2] = σx * pdf(LogNormal(log(K_sx)+ μ_max^2, μ_max), S) * X
            du[3] = σy * pdf(LogNormal(log(K_sy)+ q_max_y^2, q_max_y), S) * X
            du[4] = σz * pdf(LogNormal(log(K_sz)+ q_max_z^2, q_max_z), Y) * Y
        end
    
        # Diffusion term (normal)
        function normal!(du, u, p, t)
            S, X, Y = u
            μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, σs, σx, σy, K_ss, µ_s = p
            du[1] = σs * pdf(Normal(K_ss , µ_s), S) * X
            du[2] = σx * pdf(Normal(K_sx , μ_max), S) * X
            du[3] = σy * pdf(Normal(K_sy , q_max_y), S) * X
        end

        # Diffusion term (monod)
        function monod!(du, u, p, t)
            S, X, Y = u
            μ_max, K_sx, Y_xs, Y_ys, m_s, q_max_y, K_sy, σs, σx, σy, K_ss, µ_s = p
            du[1] = σs * (S / (K_ss + S)) * X
            du[2] = σx * (S / (K_sx + S)) * X
            du[3] = σy * (S / (K_sy + S)) * Y
        end

        # Project solution to stay ≥ 0
        function project!(integrator)
            integrator.u .= max.(integrator.u, 0.0)
        end
        proj_cb = DiscreteCallback((u,t,integrator) -> true, project!)

        # Define the problem
        odeprob = ODEProblem(drift!, u_0, (0.0, T), p)

        if noise == "lognormal"
            sdeprob = SDEProblem(drift!, lognormal!, u_0, (0.0, T), p)
        elseif noise == "normal"
            sdeprob = SDEProblem(drift!, normal!, u_0, (0.0, T), p)
        elseif noise == "monod"
            sdeprob = SDEProblem(drift!, monod!, u_0, (0.0, T), p)
        else
            error("Invalid noise type. Choose 'lognormal', 'normal', or 'monod'.")
        end

        # Solve the problem
        sdesol = solve(sdeprob, EM(), dt = dt, saveat = dt, callback = proj_cb)
        odesol = solve(odeprob, Tsit5(), dt = dt, saveat = dt)

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
            sdesolutions[i] = kinetics(params, T, u0, noise, dt)[1] # Store SDE solutions
        end
        odesolution = kinetics(params, T, u0, noise, dt)[2] # Store ODE solution
        
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

end # end the module SDEutils.jl