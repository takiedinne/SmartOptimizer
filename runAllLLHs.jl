include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using Distributions
using CSV
using DataFrames
using Plots
using Serialization


#define the range of the expirement 
replication_nbr_array = [3,5,10]
max_iteration_nbr_array = [1000]

initial_x = rand(1:20,10)
deterministic_method_list = [ HookeAndJeeves(), NelderMead(),
                             ParticleSwarm(), SimulatedAnnealing(), TabuSearch(),
                             GeneticAlgorithm(),
                              ]

HH_method_list = [ϵGreedy(), MarkovChainHH(), TabuSearchHH()]
Stochastic_method_list = [StochasticComparison(), StochasticRuler(), GeneratingSetSearcher(),
                          COMPASS_Searcher(), SimulatedAnnealingSO()  ]

df_result = DataFrame( method_name=[], replication_nbr=[], 
                        max_iteration_nbr = [], duration=[], x=[], fit=[], nbrSim=[],
                         real_fit=[], trace = [])

for max_iteration_nbr in max_iteration_nbr_array
    plot(xlabel = "iterations", ylabel = "f(x)")
    opt = Options(max_iterations = max_iteration_nbr)
    for replication_nbr in replication_nbr_array
        p = Problem(sim_GG1K, false, 10, upper = ones(10).*20, lower = ones(10),
                        initial_x = initial_x, replicationsNbr = replication_nbr)
        #deterministic list
        for m in deterministic_method_list
            res = optimize(m, p, opt)
            name = m.method_name
            duration = res[1].elapsed_time
            x = res[1].minimizer
            fit = res[1].minimum
            real_fit = mean([p.objective(x) for i in 1:100])
            nbrSim = res[2]
            his_best_fit = res[3]
            push!(df_result, (name , replication_nbr, max_iteration_nbr, duration, x,
                    fit, nbrSim, real_fit, his_best_fit))
        end
        #hyper heuristic list
        for m in HH_method_list
            res = HH_optimize(m, p, opt)
            name = m.method_name
            duration = res[1].elapsed_time
            x = res[1].minimizer
            fit = res[1].minimum
            real_fit = mean([p.objective(x) for i in 1:100])
            nbrSim = res[2]
            his_best_fit = res[3]
            push!(df_result, (name, replication_nbr, max_iteration_nbr, duration, x,
                    fit, nbrSim, real_fit, his_best_fit))
        end
    end
    # stochastic list
    p = Problem(sim_GG1K, false, 10, upper = ones(10).*20, lower = ones(10),
                initial_x = initial_x)
    for m in Stochastic_method_list
        res = optimize(m, p, opt)
        name = m.method_name
        duration = res[1].elapsed_time
        x = res[1].minimizer
        fit = res[1].minimum
        real_fit =mean([p.objective(x) for i in 1:100])
        nbrSim = res[2]
        his_best_fit = res[3]
        push!(df_result, (name, NaN, max_iteration_nbr, duration, x,
                fit, nbrSim, real_fit, his_best_fit))
    end
end
df_trace = df_result[:,[:method_name, :trace]]
select!(df_result, Not(:trace))
# save files
path = "C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\MultiQueueServer\\"

CSV.write(path * "results.csv", df_result)
serialize(path * "trace.jls", df_trace)

#trace = deserialize("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\MultiQueueServer\\trace.jls")

