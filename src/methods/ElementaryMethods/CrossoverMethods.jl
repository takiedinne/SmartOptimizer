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
    parents, fit_parents, nbrOfSim = get_solution_from_archive(HHState.archive, problem, 2, order=:roulette)
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
struct BinomialCrossover <: Crossover
    method_name::String
    rate::Float64
end
BinomialCrossover(;rate=0.5)= BinomialCrossover("Binomial Crossover", rate)

function update_state!(method::BinomialCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    state.x = copy(state.parent1)
    for i in 1 : length(state.x)
        if rand() < method.rate
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

# Interpolation CrossOver in other references it called heuristic crossover
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

struct ExponetialCrossover <: Crossover
    method_name::String
    μ::Float64
end
ExponetialCrossover(;mu=0.5)= ExponetialCrossover("Exponential Crossover", mu)

function update_state!(method::ExponetialCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    state.x = copy(state.parent1)
    l = problem.dimension
    j = rand(1:l)
    switch = true
    for i in (((1:l).+j.-2).%l).+1
        i == j && continue
        if switch && rand() <= method.μ
            state.x[i] = state.parent1[i]
        else
            switch = false
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

struct KPointCrossover <: Crossover
    method_name::String
    k::Integer #nbr of crossover point 
end
KPointCrossover(;k::Integer=2) = KPointCrossover("$k-point Crossover", k)
#

function update_state!(method::KPointCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    state.x = copy(state.parent1)
    l = problem.dimension
    crossover_points =sample(1:l, method.k, replace = false)  # in the best situation we will have k different points 
    push!(crossover_points, l) # we add the last point 
    sort!(crossover_points)
    current_parent = state.parent1
    current_index = 1
    for i in crossover_points
        state.x[current_index: i] = current_parent[current_index: i]
        current_parent = (current_parent == state.parent1) ? state.parent2 : state.parent1
        current_index = i+1
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

# averaging is the same as interpolation but we take λ = 0.5


struct FlatCrossover <: Crossover
    method_name::String
end
FlatCrossover() = FlatCrossover("Flat Crossover")

function update_state!(method::FlatCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    state.x = copy(state.parent1)
    l = problem.dimension
    for i in 1:l
        a, b =  (state.parent1[i] > state.parent2[i]) ? (state.parent2[i], state.parent1[i]) : (state.parent1[i], state.parent2[i] )
        state.x[i] = rand(a:b)
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

""" permuation crossover """

struct OrderBasedCrossover <: Crossover
    method_name::String
end 
OrderBasedCrossover() = OrderBasedCrossover("order based Crossover")

function update_state!(method::OrderBasedCrossover, problem::Problem{T}, iteration::Int, state::CrossoverState) where {T}
    # this methods is applied only to permutation based solution
    state.x = zero(state.parent1)
    l = problem.dimension
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    # Swap
    state.x[from:to] = state.parent2[from:to]
    # Fill in from parents
    k = to+1 > l ? 1 : to+1 #child1 index
    j = to+1 > l ? 1 : to+1 #child2 index
    for i in vcat(to+1:l,1:from-1)
        while in(state.parent1[k],state.x)
            k = k+1 > l ? 1 : k+1
        end
        state.x[i] = state.parent1[k]
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