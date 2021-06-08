include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using DataFrames
using CSV
p = Problem(sim_GG1K,false,10, upper=Integer.(ones(10).*20), lower= Integer.(ones(10)))
m= TabuSearch() 
#optimize(m, p)

archive = DataFrame(x= [[15,9,15,11,4,14,15,14,16,1], [20,9,15,11,4,14,15,14,16,1]], 
fit= [1200.0,1500.0])
optimize(m,p)

s = Main.SmartOptimizer.create_state_for_HH(m, p, archive)

res = Main.SmartOptimizer.update_state!(m,p,1,s[1])