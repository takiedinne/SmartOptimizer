include("../src/SmartOptimizer.jl")
using Main.SmartOptimizer
using Distributions
using CSV
using DataFrames
using Plots

# define the function
f1(X) = -1*(0.4*X[1]-5)^2 - 2*(0.4*X[2]-17.2)^2 +7
f2(X) = -1*(0.4*X[1]-12)^2 - (0.4*X[2]-4)^2 +4
f(X)  = max(f1(X), f2(X), 0)
h(X)  = -1*(f(X) + rand(Normal(0,50))) 
#define the range of the expirement 
replication_nbr_array = [3, 5, 10, 20]
max_iteration_nbr_array = [500, 1000, 10000, 50000]
σ_array = [1, 3, 5, 10, 50]


initial_x = rand(0:49,2)
deterministic_method_list = [GeneticAlgorithm(), HookeAndJeeves(), NelderMead(),
                             ParticleSwarm(), SimulatedAnnealing(), TabuSearch() ]

HH_method_list = [ϵGreedy(), MarkovChainHH(), TabuSearchHH()]
Stochastic_method_list = [StochasticComparison(), StochasticRuler(), GeneratingSetSearcher(),
                          COMPASS_Searcher(), SimulatedAnnealingSO()  ]

df_result = DataFrame( method_name=[], σ=[], replication_nbr=[], 
                        max_iteration_nbr = [], duration=[], x=[], fit=[], nbrSim=[], real_fit=[])

for σ in σ_array
    h(X)  = -1*(f(X) + rand(Normal(0,σ)))  # it's the objective function
    for max_iteration_nbr in max_iteration_nbr_array
        opt = Options(max_iterations = max_iteration_nbr)
        for replication_nbr in replication_nbr_array
            p = Problem(h, false, 2, upper = [49,49], lower = [0,0],
                            initial_x = initial_x, replicationsNbr = replication_nbr)
            #deterministic list
            for m in deterministic_method_list
                res = optimize(m, p, opt)
                name = m.method_name
                duration = res[1].elapsed_time
                x = res[1].minimizer
                fit = res[1].minimum
                real_fit = -1*f(x)
                nbrSim = res[2]
                push!(df_result, (name, σ, replication_nbr, max_iteration_nbr, duration, x,
                        fit, nbrSim, real_fit))
            end
            #hyper heuristic list
            for m in HH_method_list
                res = HH_optimize(m, p, opt)
                name = m.method_name
                duration = res[1].elapsed_time
                x = res[1].minimizer
                fit = res[1].minimum
                real_fit = -1*f(x)
                nbrSim = res[2]
                push!(df_result, (name, σ, replication_nbr, max_iteration_nbr, duration, x,
                        fit, nbrSim, real_fit))
            end
        end
        # stochastic list
        p = Problem(h, false, 2, upper = [49,49], lower = [0,0],
                            initial_x = initial_x)
        for m in Stochastic_method_list
            res = optimize(m, p, opt)
            name = m.method_name
            duration = res[1].elapsed_time
            x = res[1].minimizer
            fit = res[1].minimum
            real_fit = -1*f(x)
            nbrSim = res[2]
            push!(df_result, (name, σ, NaN, max_iteration_nbr, duration, x,
                    fit, nbrSim, real_fit))
        end
    end
    
end
CSV.write("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\TwoHillsProblem\\results.csv", df_result)

