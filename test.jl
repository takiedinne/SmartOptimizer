include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer

p = Problem(sim_GG1K,false,10, upper=Integer.(ones(10).*20), lower= Integer.(ones(10)))
#=
m=MarkovChainHH(MA=Main.SmartOptimizer.NaiveAcceptance, LF = Main.SmartOptimizer.QLearning_LM())
r = HH_optimize(m,p)=#

m =TabuSearchHH()
HH_optimize(m, p)