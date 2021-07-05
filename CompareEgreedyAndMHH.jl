include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using DataFrames
using CSV

p = Problem(sim_GG1K,false,10, upper=Integer.(ones(10).*20), lower= Integer.(ones(10)))
x=p.x_initial
methods= DataFrame( abv=[], method=[], duration=[], x=[], fit=[], nbrSim=[] )

#we do the first experiment with the fix rexard panishement
push!(methods, ["MHH_FixR_OI", MarkovChainHH(MA=Main.SmartOptimizer.OnlyImprovement), 0, x, Inf, 0])
push!(methods, ["MHH_FixR_AM", MarkovChainHH(MA=Main.SmartOptimizer.AllMoves), 0, x, Inf, 0])
push!(methods, ["MHH_FixR_NM", MarkovChainHH(MA=Main.SmartOptimizer.NaiveAcceptance), 0, x, Inf, 0])
push!(methods, ["MHH_SARSA_OI", MarkovChainHH(MA=Main.SmartOptimizer.OnlyImprovement, LF =Main.SmartOptimizer.SARSA_LM() ), 0, x, Inf, 0])
push!(methods, ["MHH_SARSA_AM", MarkovChainHH(MA=Main.SmartOptimizer.AllMoves, LF =Main.SmartOptimizer.SARSA_LM()), 0, x, Inf, 0])
push!(methods, ["MHH_SARSA_NM", MarkovChainHH(MA=Main.SmartOptimizer.NaiveAcceptance, LF =Main.SmartOptimizer.SARSA_LM()), 0, x, Inf, 0])
push!(methods, ["MHH_QL_OI", MarkovChainHH(MA=Main.SmartOptimizer.OnlyImprovement, LF =Main.SmartOptimizer.QLearning_LM()), 0, x, Inf, 0])
push!(methods, ["MHH_QL_AM", MarkovChainHH(MA=Main.SmartOptimizer.AllMoves, LF =Main.SmartOptimizer.QLearning_LM()), 0, x, Inf, 0])
push!(methods, ["MHH_QL_NM", MarkovChainHH(MA=Main.SmartOptimizer.NaiveAcceptance, LF =Main.SmartOptimizer.QLearning_LM()), 0, x, Inf, 0])

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

CSV.write("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\HyperHeuristic\\MHH1000Iter.csv", methods)
