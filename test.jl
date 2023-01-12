include("simulations/GG1K_simulation.jl")
include("src/SmartOptimizer.jl")

using Main.SmartOptimizer


# define the Problem
obj_fun = sim_GG1K
nbr_variables = 10
upper = ones(Int, 10) .* 20
lower = ones(Int, 10)
initial_x = [rand( lower[i]:upper[i] ) for i in 1:nbr_variables]

p = Problem(obj_fun, false, nbr_variables, upper = upper, lower = lower)
# the method 
m = GeneticAlgorithm()

res = optimize(m, p)

