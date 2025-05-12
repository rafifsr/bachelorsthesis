using DifferentialEquations

# Define parameters
k1 = 1.0       # binding rate
k_1 = 0.5      # unbinding rate
k2 = 0.2       # product formation rate

# Total enzyme concentration
E_total = 1.0  

# Define the ODE system
function michaelis_menten!(du, u, p, t)
    E, S, ES, P = u
    du[1] = -k1 * E * S + (k_1 + k2) * ES         # d[E]/dt
    du[2] = -k1 * E * S + k_1 * ES                 # d[S]/dt
    du[3] = k1 * E * S - (k_1 + k2) * ES           # d[ES]/dt
    du[4] = k2 * ES                               # d[P]/dt
end

# Initial conditions:  ,  ,  ,  
u0 = [E_total, 10.0, 0.0, 0.0]

# Time span
tspan = (0.0, 50.0)

# Define the ODE problem
prob = ODEProblem(michaelis_menten!, u0, tspan)

# Solve the ODE
sol = solve(prob)

# Plotting (optional)
using Plots
plot(sol, labels=["[E]" "[S]" "[ES]" "[P]"], xlabel="Time", ylabel="Concentration", lw=2)
plot!(title="Michaelis-Menten Kinetics", legend=:topright, grid=true, fontfamily="Computer Modern")