include("src/SmartOptimizer.jl")
include("simulations/one_layer_with_gap_GA_sand.jl")
using Main.SmartOptimizer

p = Problem(fobj_horizontal, false, nvars, upper = round.(Int, upper), lower = round.(Int,lower))

m = TabuSearchHH()
HH_optimize(m, p, Options(max_iterations=1000))