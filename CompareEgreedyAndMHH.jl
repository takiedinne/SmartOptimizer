include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using DataFrames
using CSV

p = Problem(sim_GG1K,false,10, upper=Integer.(ones(10).*20), lower= Integer.(ones(10)))
x=p.x_initial
methods= DataFrame( abv=[], method=[], duration=[], x=[], fit=[], nbrSim=[] )

#we do the first experiment with the fix rexard panishement
push!(methods, ["EGreedy_FixR_OI",ϵGreedy(MA=Main.SmartOptimizer.OnlyImprovement), 0, x, Inf, 0])
push!(methods, ["EGreedy_FixR_AM", ϵGreedy(MA=Main.SmartOptimizer.AllMoves), 0, x, Inf, 0])
push!(methods, ["EGreedy_FixR_NM", ϵGreedy(MA=Main.SmartOptimizer.NaiveAcceptance), 0, x, Inf, 0])
push!(methods, ["EGreedy_SARSA_OI", ϵGreedy(MA=Main.SmartOptimizer.OnlyImprovement, LM =Main.SmartOptimizer.SARSA_LM() ), 0, x, Inf, 0])
push!(methods, ["EGreedy_SARSA_AM", ϵGreedy(MA=Main.SmartOptimizer.AllMoves, LM =Main.SmartOptimizer.SARSA_LM()), 0, x, Inf, 0])
push!(methods, ["EGreedy_SARSA_NM", ϵGreedy(MA=Main.SmartOptimizer.NaiveAcceptance, LM =Main.SmartOptimizer.SARSA_LM()), 0, x, Inf, 0])
push!(methods, ["EGreedy_QL_OI", ϵGreedy(MA=Main.SmartOptimizer.OnlyImprovement, LM =Main.SmartOptimizer.QLearning_LM()), 0, x, Inf, 0])
push!(methods, ["EGreedy_QL_AM", ϵGreedy(MA=Main.SmartOptimizer.AllMoves, LM =Main.SmartOptimizer.QLearning_LM()), 0, x, Inf, 0])
push!(methods, ["EGreedy_QL_NM", ϵGreedy(MA=Main.SmartOptimizer.NaiveAcceptance, LM =Main.SmartOptimizer.QLearning_LM()), 0, x, Inf, 0])

for i in 1:nrow(methods)
    method= methods[i, :method]
    res=0
    try
        res = HH_optimize(method,p)
    catch e
        println("the ", method.method_name, " is interrupted...")
        msg = sprint(showerror, e)
        println(msg)
        continue
    end
    if res != 0
        methods[i, :duration]=res[1].elapsed_time
        methods[i, :x]=res[1].minimizer
        methods[i, :fit]=res[1].minimum
        methods[i, :nbrSim]=res[2]
    end
    
end 
select!(methods, Not(:method))

CSV.write("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\HyperHeuristic\\EpsilonGreedy\\EGreedy500Iter.csv", methods)
