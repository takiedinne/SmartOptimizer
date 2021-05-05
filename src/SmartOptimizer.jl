module SmartOptimizer
    using LinearAlgebra
    using Random
    using DataFrames
    using Evolutionary
    using Statistics
    using Distributions
    using Combinatorics

    include("types.jl")
    include("api.jl")
    include("methods/DiscreteSimulationOptimizationMethods/GeneratingSetSearch.jl")
    include("methods/DiscreteSimulationOptimizationMethods/Compass.jl")
    include("methods/DiscreteSimulationOptimizationMethods/GeneticAlgorithm.jl")
    include("methods/DiscreteSimulationOptimizationMethods/HookeJeeves.jl")
    include("methods/DiscreteSimulationOptimizationMethods/NelderMead.jl")
    include("methods/DiscreteSimulationOptimizationMethods/OCBA.jl")
    include("methods/DiscreteSimulationOptimizationMethods/ParticleSwarm.jl")
    include("methods/DiscreteSimulationOptimizationMethods/SimulatedAnnealing.jl")
    include("methods/DiscreteSimulationOptimizationMethods/StochasticComparison.jl")
    include("methods/DiscreteSimulationOptimizationMethods/StochasticRuler.jl")
    include("methods/DiscreteSimulationOptimizationMethods/TabuSearch.jl")
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
        TabuSearch
end
