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
mutable struct GA <: LowLevelHeuristic
    method_name::String
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::Function
    crossover::Function
    mutation::Function

    GA(; populationSize::Int=10, crossoverRate::Float64=0.8, mutationRate::Float64=0.1,
        ɛ::Real=0.2, epsilon::Real=ɛ,
        selection::Function = roulette,
        crossover::Function = twopoint, mutation::Function = mutation_fun) =
        new("Genetic Algorithm", populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation)
end


mutable struct GAState{T,IT} <: State
    N::Int #dimension
    eliteSize::Int # fraction or Number of instances used in the cross over step
    x_fitness::T # min fitness
    fitpop::Vector{T} # all population's fitness
    x::IT # the best solution
    population::Vector{IT}
end

value(s::GAState) = s.x_fitness
minimizer(s::GAState) = s.x

function initial_population(method::GA, problem::Problem{T}) where {T} 
    size_population=method.populationSize
    population= []
    for i in 1:size_population
        x=copy(problem.x_initial)
        random_x!(x,problem.dimension, upper=problem.upper, lower=problem.lower)
        push!(population,x)
    end
    return population
end
#Initialization of GA algorithm state"""
function initial_state(method::GA, problem::Problem{T}) where {T<:Number}
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
function update_state!(method::GA, problem::Problem{T}, iteration::Int, state::GAState) where {T}
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
            mutation(offspring[i],problem)
        end
    end

    # Create new generation & evaluate it
    population=offspring
    for i in 1:populationSize
        state.fitpop[i] = problem.objective(population[i])
    end
    
    # find the best individual
    minfit, fitidx = findmin(state.fitpop)
    state.x = population[fitidx]
    state.x_fitness = state.fitpop[fitidx]
    #return the best values and it is the current
    
    state.x , state.x_fitness, method.populationSize
end
function create_state_for_HH(method::GA, problem::Problem, HHState::HH_State)
    nbrSim = 0
    N = problem.dimension
    # setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)
    # it is batter to use get solutions from archive function if we want
    population = DataFrame(x= [HHState.x], fit = [HHState.x_fit])
    x, fit, nbrSim = get_solution_from_archive(HHState.archive, problem, method.populationSize-1 )
    append!(population, (x = x, fit = fit))
    # setup initial state
    GAState(N, eliteSize, HHState.x_fit, fitness, copy(HHState.x), population), nbrSim
end
