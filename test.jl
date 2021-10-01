#include("simulations/GG1K_simulation.jl")
include("src/SmartOptimizer.jl")
using Main.SmartOptimizer
#using DataFrames
#include("simulations/InventoryRoutingProblem.jl")
#include("simulations/MTO_products.jl")
#=
X = [rand(1:20,10) for i in 1: 100]
std_deviation = []
for x in X
    y = []
    for i in 1:1000
        push!(y, sim_GG1K(x))
    end
    push!(std_deviation, std(y))
end
final_σ = mean(std_deviation)
=#

# define the function
f1(X) = -1*(0.4*X[1]-5)^2 - 2*(0.4*X[2]-17.2)^2 +7
f2(X) = -1*(0.4*X[1]-12)^2 - (0.4*X[2]-4)^2 +4
f(X)  = max(f1(X), f2(X), 0)
h(X)  = -1*(f(X) + rand(Normal(0,1))) 
using Distributions
#p = Problem(sim_GG1K, false, 10, upper = ones(10).*20, lower = ones(10))
p = Problem(f, false, 2, upper = ones(2).*49, lower = zeros(2), replicationsNbr = 50, initial_x =[7,41])
m =ϵGreedy(LM = Main.SmartOptimizer.LearningAutomata(),
            MA = Main.SmartOptimizer.NaiveAcceptance())
HH_optimize(m, p, Options(max_iterations = 10000))
f([11,43])
p.x_initial= [7,41]
