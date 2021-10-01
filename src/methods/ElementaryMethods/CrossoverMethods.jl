"""
crossover methods given two solution produce a new solution (child) by combining the parents
these crossover methods are implemented as the way explained in [1]

[1]: Kochenderfer, M. J., Wheeler, T. A., Wray, K. H., Gane, G. P.,
        Horabin, I. S., & Lewis, B. N. (2021). 
        Algorithms for decision making. In The proceedings of the programmed
        learning conference.
        https://algorithmsbook.com/files/dm.pdf

P.S : all this methods will use only with Hyper heuristic, so they will be generaly invoked one time 
"""

abstract type Crossover <: LowLevelHeuristic end

mutable struct CrossoverState{T} <: State
    x::AbstractArray{T,1} # the best solution here it's the child
    f_x # the fitness of the new solution
    parent1::AbstractArray{T,1} 
    parent2::AbstractArray{T,1} 
    f_parent1
    f_parent2
end 

function create_state_for_HH(method::Crossover, problem::Problem{T}, HHState::HH_State) where {T<:Number}
    # get two solution to be parents from the archive
    parent1, fit_parent1 = HHState.x, HHState.x_fit # take the curent solution 
    # we need another solution  
    parents, fit_parents, nbrOfSim = get_solution_from_archive(HHState.archive, problem, 2)
    parent2, fit_parent2 = (parents[1] != parent1) ? (parents[1], fit_parents[1]) : (parents[2], fit_parents[2])
    CrossoverState(typeof(parent1)(), # the child initially not defined  
                NaN, # fitness of the child initially Not a number
                parent1, # first parent
                parent2, # second parent
                fit_parent1, # fitness parent1
                fit_parent2 # fitness parent2
                ), nbrOfSim 
end

# single Point
struct SinglePointCrossover <:  Crossover
    method_name::String
end
SinglePointCrossover()= SinglePointCrossover("Single Point Crossover")

function update_state!(method::SinglePointCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    i = rand(1:length(state.parent1))
    state.x = vcat(state.parent1[1:i],state.parent2[i+1:end])
    state.f_x = problem.objective(state.x)
    # here we get two parents here the child will be parents 2 and parent 2 will be parent 1
    # this step not really necessary but if the hyper heuristic use this Method for several iteration
    # it'll be useful
    state.parent1 = state.parent2
    state.f_parent1 = state.f_parent2
    state.parent2 = state.x
    state.f_parent2 = state.f_x

    state.x, state.f_x, 1# we've invoked the similation only one time
end

# Two Point 
struct TwoPointCrossover <: Crossover
    method_name::String
end
TwoPointCrossover()= TwoPointCrossover("Two Point Crossover")

function update_state!(method::TwoPointCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    i, j = rand(1:length(state.parent1), 2)
    if i > j
        (i,j) = (j,i)
    end

    state.x = vcat(state.parent1[1:i],state.parent2[i+1:j], state.parent1[j+1:end])
    state.f_x = problem.objective(state.x)
    # here we get two parents here the child will be parents 2 and parent 2 will be parent 1
    # this step not really necessary but if the hyper heuristic use this Method for several iteration
    # it'll be useful
    state.parent1 = state.parent2
    state.f_parent1 = state.f_parent2
    state.parent2 = state.x
    state.f_parent2 = state.f_x

    state.x, state.f_x, 1# we've invoked the similation only one time
end

# Uniform Crossover 
struct UniformCrossover <: Crossover
    method_name::String
end
UniformCrossover()= UniformCrossover("Uniform Crossover")

function update_state!(method::UniformCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    state.x = copy(state.parent1)
    for i in 1 : length(state.x)
        if rand() < 0.5
            state.x[i] = state.parent2[i]
        end
    end
    state.f_x = problem.objective(state.x)
    # here we get two parents here the child will be parents 2 and parent 2 will be parent 1
    # this step not really necessary but if the hyper heuristic use this Method for several iteration
    # it'll be useful
    state.parent1 = state.parent2
    state.f_parent1 = state.f_parent2
    state.parent2 = state.x
    state.f_parent2 = state.f_x

    state.x, state.f_x, 1 # we've invoked the similation only one time
end

# Interpolation CrossOver
struct InterpolationCrossover <: Crossover
    method_name::String
    λ # interpolation parameter
end
InterpolationCrossover(;lambda = 0.5)= InterpolationCrossover("Interpolation Crossover", lambda)

function update_state!(method::InterpolationCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    state.x = copy(state.parent1)
    for i in 1 : length(state.x)
        state.x[i] = round(state.parent1[i] * (1 - method.λ) + state.parent2[i] * method.λ)
    end
    state.f_x = problem.objective(state.x)
    # here we get two parents here the child will be parents 2 and parent 2 will be parent 1
    # this step not really necessary but if the hyper heuristic use this Method for several iteration
    # it'll be useful
    state.parent1 = state.parent2
    state.f_parent1 = state.f_parent2
    state.parent2 = state.x
    state.f_parent2 = state.f_x

    state.x, state.f_x, 1 # we've invoked the similation only one time
end

