module SmartOptimizer
    
    using LinearAlgebra
    using Random
    using DataFrames
    using Evolutionary
    using Statistics
    using Distributions
    using Combinatorics
    using Plots, StatsPlots
    using Distances

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
    include("methods/DiscreteSimulationOptimizationMethods/StochasticComparison.jl")
    include("methods/DiscreteSimulationOptimizationMethods/StochasticRuler.jl")
    include("methods/DiscreteSimulationOptimizationMethods/TabuSearch.jl")
    #the hyper heuristic methods
    include("HyperHeuristics/moveAcceptance.jl")
    include("HyperHeuristics/LearningFunctions.jl")
    include("HyperHeuristics/Epsilon-greedy.jl")
    include("HyperHeuristics/MarkovChainHH.jl")
    export
        optimize,
        Problem,
        Options,
        GeneratingSetSearcher,
        COMPASS_Searcher,
        GA,
        HookeAndJeeves,
        NelderMead,
        OCBA,
        ParticleSwarm,
        SimulatedAnnealing,
        StochasticComparison,
        StochasticRuler,
        TabuSearch,
        #hyper heuristic
        HH_optimize,
        ϵGreedy,
        MarkovChainHH
end
