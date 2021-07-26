include("src/SmartOptimizer.jl")
#include("simulations/GG1K_simulation.jl")
include("simulations/BrainFuck/Brainfuck.jl")
include("simulations/BrainFuck/fitnessFunctions.jl")
using Main.Brainfuck
using Main.SmartOptimizer

p = Problem(fitnessStr, false, 150, upper= ones(150).*8, lower=ones(150))
m = TabuSearchHH(es=10)
HH_optimize(m, p)