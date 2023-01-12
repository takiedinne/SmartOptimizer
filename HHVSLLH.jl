include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using DataFrames
using CSV


# define the Problem
obj_fun = sim_GG1K
nbr_variables = 10
upper = ones(Int, 10) .* 20
lower = ones(Int, 10)
initial_x = [rand( lower[i]:upper[i] ) for i in 1:nbr_variables]

#define the range of the expirement 
replication_nbr_array = [#= 3, 5, =# 5]
max_iteration_nbr_array = [5]

replication_nbr = 5
run_nbr = 1
max_iteration_nbr = 1000

function deterministic_methods_experiment()
    deterministic_method_list = [ HookeAndJeeves(), NelderMead(),
    ParticleSwarm(), SimulatedAnnealing(), TabuSearch(),
    GeneticAlgorithm(), AntColonySearcher()
   ]

    df_result = DataFrame( method_name=[], move_acceptance=[], learning_mechanism = [], replication_nbr=[], 
                        max_iteration_nbr = [], duration=[], best_x=[], best_fit=[], avg_fit=[],
                        best_real_fit=[], avg_real_fit=[], nbrSim=[])
    for m in deterministic_method_list
        println("working with ", m.method_name)
        opt = Options(max_iterations = max_iteration_nbr )
        p = Problem(obj_fun, false, nbr_variables, upper = upper, lower = lower,
                            initial_x = initial_x, replicationsNbr = replication_nbr)
        duration = 0
        best_x = [] # the best solution
        best_fit = Inf
        best_real_fit = Inf
        avg_fit = 0 # avg_fit
        avg_nbr_sim = 0
        avg_real_fit = 0

        res = optimize(m, p, opt)
        duration += res[1].elapsed_time
        x = res[1].minimizer
        fit = res[1].minimum
        real_fit = mean([p.objective(x) for i in 1:100])
        avg_real_fit += real_fit
        if fit < best_fit
            best_x = x
            best_fit = fit
            best_real_fit = real_fit
        end
        avg_fit += res[1].minimum
        avg_nbr_sim += res[2]
        #histo_best_fit = res[3]
            
        name = m.method_name
        duration /= run_nbr
        avg_nbr_sim /= run_nbr
        avg_fit /= run_nbr
        avg_real_fit /= run_nbr
        
        push!(df_result, (name ,"", "", replication_nbr, max_iteration_nbr, duration, best_x,
                best_fit,avg_fit, best_real_fit, avg_real_fit, avg_nbr_sim))
    end

    CSV.write("results/deterministicMethods.csv", df_result)
end

function stochastic_methods_experiment()
    Stochastic_method_list = [ StochasticComparison(), StochasticRuler(), GeneratingSetSearcher(), 
                          COMPASS_Searcher() , SimulatedAnnealingSO() ]

    df_result = DataFrame( method_name=[], move_acceptance=[], learning_mechanism = [], replication_nbr=[], 
                        max_iteration_nbr = [], duration=[], best_x=[], best_fit=[], avg_fit=[],
                        best_real_fit=[], avg_real_fit=[], nbrSim=[])

    for m in Stochastic_method_list
        println("working with ", m.method_name)
        opt = Options(max_iterations = max_iteration_nbr)
        p = Problem(obj_fun, false, nbr_variables, upper = upper, lower = lower,
                            initial_x = initial_x)

        duration = 0
        best_x = [] # the best solution
        best_fit = Inf
        best_real_fit = Inf
        avg_fit = 0 # avg_fit
        avg_nbr_sim = 0
        avg_real_fit = 0

        res = optimize(m, p, opt)
        duration += res[1].elapsed_time
        x = res[1].minimizer
        fit = res[1].minimum
        real_fit = mean([p.objective(x) for i in 1:100])
        avg_real_fit += real_fit
        if fit < best_fit
            best_x = x
            best_fit = fit
            best_real_fit = real_fit
        end
        avg_fit += res[1].minimum
        avg_nbr_sim += res[2]
            
        name = m.method_name
        duration /= run_nbr
        avg_nbr_sim /= run_nbr
        avg_fit /= run_nbr
        avg_real_fit /= run_nbr
        
        push!(df_result, (name ,"", "", NaN, max_iteration_nbr, duration, best_x,
                best_fit,avg_fit, best_real_fit, avg_real_fit, avg_nbr_sim))
    end

    CSV.write("results/stochasticMethods.csv", df_result)
end

function HH_methods_experiment()
    learing_mechanism_list = [reward_punish_LM(), SARSA_LM(), QLearning_LM(), LearningAutomata()]
    move_acceptance_list = [OnlyImprovement(), AllMoves(), NaiveAcceptance(), ILTA()]
    HH_method_list = []

    for ma in move_acceptance_list
        for lm in learing_mechanism_list
            append!(HH_method_list, [ϵGreedy(LM = lm, MA = ma), MarkovChainHH(LM = lm, MA = ma)])
        end
        append!(HH_method_list, [TabuSearchHH(MA = ma)])
    end

    df_result = DataFrame( method_name=[], move_acceptance=[], learning_mechanism = [], replication_nbr=[], 
                        max_iteration_nbr = [], duration=[], best_x=[], best_fit=[], avg_fit=[],
                        best_real_fit=[], avg_real_fit=[], nbrSim=[])

    for m in HH_method_list
        println("working with ", m.method_name)
        opt = Options(max_iterations = max_iteration_nbr)
        p = Problem(obj_fun, false, nbr_variables, upper = upper, lower = lower,
                            initial_x = initial_x)

        
        duration = 0
        best_x = [] # the best solution
        best_fit = Inf
        best_real_fit = Inf
        avg_fit = 0 # avg_fit
        avg_nbr_sim = 0
        avg_real_fit = 0
        for i in 1:run_nbr
            res = HH_optimize(m, p, opt)
            duration += res[1].elapsed_time
            x = res[1].minimizer
            fit = res[1].minimum
            real_fit = mean([p.objective(x) for i in 1:100])
            avg_real_fit += real_fit
            if fit < best_fit
                best_x = x
                best_fit = fit
                best_real_fit = real_fit
            end
            avg_fit += res[1].minimum
            avg_nbr_sim += res[2]
            #histo_best_fit = res[3]
        end
        name = m.method_name
        move_acceptance = m.moveAcceptance.method_name
        learning_mechanism = m.method_name != "Tabu Search Hyper Heuristic" ? m.learningMechanism.method_name : ""
        duration /= run_nbr
        avg_nbr_sim /= run_nbr
        avg_fit /= run_nbr
        avg_real_fit /= run_nbr
        push!(df_result, (name, move_acceptance, learning_mechanism,
                        replication_nbr, max_iteration_nbr, duration, best_x,
                        best_fit,avg_fit, best_real_fit, avg_real_fit, avg_nbr_sim))
    
    end

    CSV.write("results/stochasticMethods.csv", df_result)
end
stochastic_methods_experiment()






