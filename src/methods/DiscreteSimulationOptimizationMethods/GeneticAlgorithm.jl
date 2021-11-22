function mutation_fun(x,problem::Problem{T}) where {T}
    place_to_mutated=rand(1:length(x))
    new_x=copy(x)

    upper = problem.upper
    lower = problem.lower
    if length(upper) == 0 && length(lower) == 0
        new_x[place_to_mutated]=rand(T,1)
    elseif length(upper) == 0
        new_x[place_to_mutated] =lower[place_to_mutated] +  abs(rand(T,1))
    elseif length(lower) == 0
        new_x[place_to_mutated]=upper[place_to_mutated] -  abs(rand(T,1))
    else
        new_x[place_to_mutated]=rand(lower[place_to_mutated]:upper[place_to_mutated])
    end
    new_x
end
"""
    domainrange(valrange, m = 20)
Returns an in-place real valued mutation function that performs the BGA mutation scheme with the mutation range `valrange` and the mutation probability `1/m` [^1].
"""
function mutation_domainrange(x, problem::Problem; m=20)
    t1 = typeof(x)
    prob = 1.0 / m
    valrange =  -1 .* problem.lower .+ problem.upper
    function mutation(recombinant::T) where {T <: AbstractVector}
        recombinant = Float64.(recombinant)
        d = length(recombinant)
        @assert length(valrange) == d "Range matrix must have $(d) columns"
        δ = zeros(m)
        for i in 1:length(recombinant)
            for j in 1:m
                δ[j] = (rand() < prob) ? δ[j] = 2.0^(-j) : 0.0
            end
            if rand() > 0.5
                recombinant[i] += sum(δ)*valrange[i]
            else
                recombinant[i] -= sum(δ)*valrange[i]
            end
        end
        x = t1(round.(recombinant .+ x))
        check_in_bounds(problem.upper, problem.lower, x ) 
        if t1 != typeof(x)
            println(" warning ")
        end
        return x
    end
    mutation(x)
end
mutable struct GeneticAlgorithm <: LowLevelHeuristic
    method_name::String
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::Function
    crossover::Function
    mutation::Function

    GeneticAlgorithm(; populationSize::Int=10, crossoverRate::Float64=0.8, mutationRate::Float64=0.1,
        ɛ::Real=0.2, epsilon::Real=ɛ,
        selection::Function = roulette,
        crossover::Function = twopoint, mutation::Function = mutation_fun) =
        new("Genetic Algorithm", populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation)
end


mutable struct GAState{T,IT} <: State
    N::Int #dimension
    eliteSize::Int # fraction or Number of instances used in the cross over step
    f_x::T # min fitness
    fitpop::Vector{T} # all population's fitness
    x::IT # the best solution
    population::Vector{IT}
end

value(s::GAState) = s.f_x
minimizer(s::GAState) = s.x

function initial_population(method::GeneticAlgorithm, problem::Problem{T}) where {T} 
    size_population=method.populationSize
    population= []
    for i in 1:size_population
        x=copy(problem.x_initial)
        random_x!(x,problem.dimension, upper=problem.upper, lower=problem.lower)
        push!(population,x)
    end
    return population
end
#Initialization of GeneticAlgorithm algorithm state"""
function initial_state(method::GeneticAlgorithm, problem::Problem{T}) where {T<:Number}
    N = problem.dimension
    fitness = zeros(method.populationSize)
    # setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)
    population = initial_population(method, problem)
    
    # Evaluate population fitness
    fitness = map(i -> problem.objective(i), population)
    minfit, fitidx = findmin(fitness)
    # setup initial state
    return GAState(N, eliteSize, minfit, fitness, copy(population[fitidx]), population)
end

#the main algorithm
function update_state!(method::GeneticAlgorithm, problem::Problem{T}, iteration::Int, state::GAState) where {T}
    populationSize = method.populationSize
    crossoverRate = method.crossoverRate
    mutationRate = method.mutationRate
    ɛ =method.ɛ 
    selection = method.selection
    crossover = method.crossover
    mutation = method.mutation
    population = state.population
    offspring = similar(population)#initiate a array of the same size of population with undef values
    
    # Select offspring
    selected = selection(state.fitpop, populationSize) # select  condidate solution with roulette procedure it may contain replications

    # Perform mating
    offidx = randperm(populationSize) #get random ordre of population
    offspringSize = populationSize - state.eliteSize
    
    #perform crossover 
    for i in 1:2:offspringSize
        j = (i == offspringSize) ? i-1 : i+1
        if rand() < crossoverRate
            offspring[i], offspring[j] = crossover(population[selected[offidx[i]]], population[selected[offidx[j]]])
            offspring[i], offspring[j] = Int64.(offspring[i]), Int64.(offspring[j])
        else
            offspring[i], offspring[j] = population[selected[i]], population[selected[j]]
        end
    end

    # Elitism (copy population individuals to complete offspring before they pass to the offspring & get mutated)
    fitidxs = sortperm(state.fitpop)# get the rank of each element accordding to fitness
    for i in 1:state.eliteSize
        subs = offspringSize + i
        offspring[subs] = copy(population[fitidxs[i]])
    end

    # Perform mutation
    for i in 1:offspringSize
        if rand() < mutationRate
            mutation(offspring[i],problem)
        end
    end

    # Create new generation & evaluate it
    population = offspring
    for i in 1:populationSize
        state.fitpop[i] = problem.objective(population[i])
    end
    
    # find the best individual
    minfit, fitidx = findmin(state.fitpop)
    if minfit <= state.f_x
        state.x = population[fitidx]
        state.f_x = state.fitpop[fitidx]
    end
    
    #return the best values and it is the current
    #state.x , state.f_x, method.populationSize
    population[fitidx], state.fitpop[fitidx], method.populationSize
end
function create_state_for_HH(method::GeneticAlgorithm, problem::Problem, HHState::HH_State)
    nbrSim = 0
    N = problem.dimension
    # setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)
    # it is batter to use get solutions from archive function if we want
    population = [HHState.x]
    pop_fit = [HHState.x_fit]
    x, fit, nbrSim = get_solution_from_archive(HHState.archive, problem, method.populationSize-1 )
    append!(population, x)
    append!(pop_fit, fit)
    # setup initial state
    GAState(N, eliteSize, HHState.x_fit, pop_fit, copy(HHState.x), population), nbrSim
end
