using UnPack
using Evolutionary
include("../../../simulations/GG1K_simulation.jl")
abstract type AbstractOptimizer end

mutable struct GA <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::Function
    crossover::Function
    mutation::Function

    GA(; populationSize::Int=5, crossoverRate::Float64=0.8, mutationRate::Float64=0.1,
        ɛ::Real=0.2, epsilon::Real=ɛ,
        selection::Function = roulette,
        crossover::Function = twopoint, mutation::Function = identity) =
        new(populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation)
end

population_size(method::GA) = method.populationSize
default_options(method::GA) = (iterations=10, abstol=1e-15)
summary(m::GA) = "Genetic_Algorithm[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate),ɛ=$(m.ɛ)]"
show(io::IO,m::GA) = print(io, summary(m))

abstract type AbstractOptimizerState end
mutable struct GAState{T,IT} <: AbstractOptimizerState
    N::Int
    eliteSize::Int
    fitness::T
    fitpop::Vector{T}
    fittest::IT
end

value(s::GAState) = s.fitness
minimizer(s::GAState) = s.fittest

function initial_population(method::M, bounds:: Tuple{Int, Int}, dim::Int) where {M<:AbstractOptimizer}
    size_population=population_size(method)
    population= []
    for i in 1:size_population
        push!(population,rand(bounds[1]:bounds[2],dim))
    end
    return population
end
#Initialization of GA algorithm state"""
function initial_state(method::GA, objfun::Function, population)
    T = typeof(objfun(first(population)))
    N = length(first(population))
    fitness = zeros(T, method.populationSize)

    # setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)

    # Evaluate population fitness
    fitness = map(i -> objfun(i), population)
    minfit, fitidx = findmin(fitness)

    # setup initial state
    return GAState(N, eliteSize, minfit, fitness, copy(population[fitidx]))
end

function mutation_fun(x,bounds)
    place_to_mutated=rand(1:length(x))
    new_x=copy(x)
    new_x[place_to_mutated]=rand(bounds[1]:bounds[2])
    new_x
end

#the main algorithm
function GeneticAlgorithme(objfun,  bounds:: Tuple{Int, Int}, dim::Int) 
    #initiate the GA
    method= GA(selection=Evolutionary.roulette, crossover=twopoint, mutation=mutation_fun)
    
    @unpack populationSize,crossoverRate,mutationRate,ɛ,selection,crossover,mutation = method
    
    @show population=initial_population(method, bounds, dim)
    state=initial_state(method,objfun,population)

    nbr_iter=0
    while nbr_iter<1000
        offspring = similar(population)#initiate a array of the same size of population with undef values

        # Select offspring
        selected = selection(state.fitpop, populationSize) # select  condidate solution with roulette procedure it may contain replucation

        # Perform mating
        offidx = randperm(populationSize) #get random ordre of population
        offspringSize = populationSize - state.eliteSize
        #perform crossover 
        
        for i in 1:2:offspringSize
            j = (i == offspringSize) ? i-1 : i+1
            if rand() < crossoverRate
                offspring[i], offspring[j] = crossover(population[selected[offidx[i]]], population[selected[offidx[j]]])
            else
                offspring[i], offspring[j] = population[selected[i]], population[selected[j]]
            end
        end

        # Elitism (copy population individuals to complete offspring before they pass to the offspring & get mutated)
        fitidxs = sortperm(state.fitpop)# get the rank of each element accordding to fitness
        for i in 1:state.eliteSize
            subs = offspringSize+i
            offspring[subs] = copy(population[fitidxs[i]])
        end

        # Perform mutation
        for i in 1:offspringSize
            if rand() < mutationRate
                mutation(offspring[i],bounds)
            end
        end

        # Create new generation & evaluate it
        population=offspring
        for i in 1:populationSize
            state.fitpop[i] = objfun(population[i])
        end
        
        # find the best individual
        minfit, fitidx = findmin(state.fitpop)
        state.fittest = population[fitidx]
        state.fitness = state.fitpop[fitidx]

        nbr_iter+=1
    end #end while true
    minfit, fitidx = findmin(state.fitpop)
    state.fittest , state.fitness 
end

GeneticAlgorithme(sim_GG1K,(1,5),3)