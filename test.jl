include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
p= Problem(sim_GG1K,false,3, upper=[5,5,5], lower= [1,1,1])
m= GA()
optimize(m,p) 
