"""
Mutation methods given a solution produce a new solution by mutation one or many variables
in the solution. the methdos are implemented as the way explained in [1]

[1]: Kochenderfer, M. J., Wheeler, T. A., Wray, K. H., Gane, G. P.,
        Horabin, I. S., & Lewis, B. N. (2021). 
        Algorithms for decision making. In The proceedings of the programmed
        learning conference.
        https://algorithmsbook.com/files/dm.pdf

P.S : all this methods will use only with Hyper heuristic, so they will be generaly invoked one time 
"""
abstract type MutationMethod <: LowLevelHeuristic end

mutable struct MutationState{T} <: State
    x::AbstractArray{T,1} # the best solution here it's the child
    f_x # the fitness of the new solution
end 

function create_state_for_HH(method::MutationMethod, problem::Problem{T}, HHState::HH_State) where {T<:Number}
    # get two solution to be parents from the archive
    x, fit, nbrOfSim = HHState.x, HHState.fit_x, 0
    MutationState(x, fit), nbrOfSim 
end

struct bitWiseMutation <: MutationMethod
    method_name::String
    γ #mutation rate
end
bitWiseMutation(;gamma = 0.2) = bitWiseMutation("bitWise mutation", gamma)

function update_state!(method::bitWiseMutation, problem::Problem{T}, iteration::Int, state::MutationState) where {T}
   
    state.x = [rand() < method.γ ? v + rand([-1,1]) : v for v in state.x]
    check_in_bounds(problem.upper, problem.lower, state.x)

    state.f_x = problem.objective(state.x)
    
    state.x, state.f_x, 1# we've invoked the similation only one time
end

struct GaussianMutation <: MutationMethod
    method_name::String
    σ #standard deviation 
end
GaussianMutation(;sigma = 3) = GaussianMutation("Gaussian mutation", sigma)

function update_state!(method::GaussianMutation, problem::Problem{T}, iteration::Int, state::MutationState) where {T}
   
    state.x = round.(state.x + randn(length(state.x)) * method.σ)
    check_in_bounds(problem.upper, problem.lower, state.x)

    state.f_x = problem.objective(state.x)
    
    state.x, state.f_x, 1# we've invoked the similation only one time
end

