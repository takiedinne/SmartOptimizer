module SmartOptimizer
    
    using LinearAlgebra
    using Random
    using DataFrames
    using Evolutionary
    using Statistics, StatsBase
    using Distributions
    using Combinatorics
    using Plots, StatsPlots
    using Distances
    using CSV
    using LightGraphs
    using GraphPlot

    include("types.jl")
    include("api.jl")
    include("methods/DiscreteSimulationOptimizationMethods/GeneratingSetSearch.jl")
    include("methods/DiscreteSimulationOptimizationMethods/Compass.jl")
    include("methods/DiscreteSimulationOptimizationMethods/GeneticAlgorithm.jl")
    include("methods/DiscreteSimulationOptimizationMethods/HookeJeeves.jl")
    include("methods/DiscreteSimulationOptimizationMethods/NelderMead.jl")
    include("methods/OtherMethods/OCBA.jl")
    include("methods/DiscreteSimulationOptimizationMethods/ParticleSwarm.jl")
    include("methods/DiscreteSimulationOptimizationMethods/SimulatedAnnealing.jl")
    include("methods/DiscreteSimulationOptimizationMethods/SimulatedAnnealingSO.jl")
    include("methods/DiscreteSimulationOptimizationMethods/StochasticComparison.jl")
    include("methods/DiscreteSimulationOptimizationMethods/StochasticRuler.jl")
    include("methods/DiscreteSimulationOptimizationMethods/TabuSearch.jl")
    include("methods/DiscreteSimulationOptimizationMethods/AntsColony.jl")
    #the hyper heuristic methods
    include("HyperHeuristics/moveAcceptance.jl")
    include("HyperHeuristics/LearningFunctions.jl")
    include("HyperHeuristics/Epsilon-greedy.jl")
    include("HyperHeuristics/MarkovChainHH.jl")
    include("HyperHeuristics/TabuSearchHH.jl")
    # elementary methods
    include("methods/ElementaryMethods/CrossoverMethods.jl")
    include("methods/ElementaryMethods/MutationMethods.jl")
    include("methods/ElementaryMethods/LocalSearchMethods.jl")
    include("methods/ElementaryMethods/neighbourhood_structure.jl")
    
    export
        optimize,
        Problem,
        Options,
        GeneratingSetSearcher,
        COMPASS_Searcher,
        GeneticAlgorithm,
        HookeAndJeeves,
        NelderMead,
        OCBA,
        ParticleSwarm,
        SimulatedAnnealing,
        SimulatedAnnealingSO,
        StochasticComparison,
        StochasticRuler,
        TabuSearch,
        AntColonySearcher,
        # moveAcceptance
        OnlyImprovement, AllMoves, NaiveAcceptance, ILTA,
        # learning mechanism
        reward_punish_LM, SARSA_LM, QLearning_LM, LearningAutomata,
        #hyper heuristic
        HH_optimize,
        ϵGreedy,
        MarkovChainHH,
        TabuSearchHH,
        # elementary methods
        SinglePointCrossover, TwoPointCrossover, UniformCrossover, InterpolationCrossover
end
