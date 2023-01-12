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
    x, fit, nbrOfSim = HHState.x, HHState.x_fit, 0
    MutationState(x, fit), nbrOfSim 
end

struct bitWiseMutation <: MutationMethod
    method_name::String
    γ #mutation rate
end
bitWiseMutation(;gamma = 0.2) = bitWiseMutation("bitWise mutation", gamma)
#add one or substract one to the elements (like bit wise we change the last bite for the element)
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
"""
    Integer mutation operator is taken from [1]. It's adapative mutation operator where the muatation value is depended on the current value.
    [1] G. Moghadampour, “Self-adaptive integer and decimal mutation operators for genetic algorithms,” ICEIS 2011 - Proc. 13th Int. Conf. Enterp.
         Inf. Syst., vol. 2 AIDSS, pp. 184–191, 2011, doi: 10.5220/0003494401840191.
"""
struct SelfAdaptiveMutation <: MutationMethod
    method_name::String
    γ::Float64 # mutation rate
end
SelfAdaptiveMutation(;gamma = 0.3) = SelfAdaptiveMutation("Self-Adaptive Integer mutation operator", gamma)

function update_state!(method::SelfAdaptiveMutation, problem::Problem{T}, iteration::Int, state::MutationState) where {T}
    # select the posistion where a mutation will be occured
    positions = rand( Bernoulli(method.γ), problem.dimension )
    # at least one posistion must be mutated 
    if sum(positions) == 0 
        positions[rand(1:problem.dimension)] = 1
    end
    for i in findall(x-> x , positions)
        Δ = rand(1:abs(state.x[i])) * rand([-1, 1])
        state.x[i] += Δ
    end
    check_in_bounds(problem.upper, problem.lower, state.x)
    state.f_x = problem.objective(state.x)
    
    state.x, state.f_x, 1# we've invoked the similation only one time
end

"""
    Integer mutation operator is taken from [1]. It's is basicaly implemented for mixte integer search space. hence it is implemented also in Evolutionary.jl under the name of mipmutation.
    [1] K. Deep, K. P. Singh, M. L. Kansal, and C. Mohan, “A real coded genetic algorithm for solving integer and mixed integer optimization problems,” 
            Appl. Math. Comput., vol. 212, no. 2, pp. 505–518, 2009, doi: 10.1016/j.amc.2009.02.044..
"""
struct PowerMutation <: MutationMethod
    method_name::String
    p_int::Float64
end
PowerMutation(;p_int = 4) = PowerMutation("power mutation", p_int)

function pm_mutation(x, l, u, s, d)
    x̄ = d < rand() ? x - s * (x - l) : x + s * (u - x)
    if isinteger(x̄)
        Int(x̄)
    else
        floor(Int, x̄) + (rand() > 0.5) # anpother method for rounding
    end
end

function update_state!(method::PowerMutation, problem::Problem{T}, iteration::Int, state::MutationState) where {T}
    @assert length(problem.upper) == length(problem.lower) == problem.dimension "the problem upper and lower bounds are incorrect... "
    # select the posistion where a mutation will be occured
    #positions = rand( Bernoulli(method.γ), problem.dimension )
    
    u = rand()
    S = ones(problem.dimension) .* (u ^ (1 / method.p_int)) # random var following power distribution
    D = (state.x - problem.lower) ./ (problem.upper - problem.lower)
    broadcast!((x,l,u,s,d)->pm_mutation(x,l,u,s,d),
                state.x, state.x, problem.lower, problem.upper, S, D)
    
    state.x, state.f_x, 1# we've invoked the similation only one time
end

"""
    non unifororm mutation from [1]
    [1] J. A. Joines, C. T. Culbreth, and R. E. King, “Manufacturing cell design: An integer programming model employing genetic algorithms,”
         IIE Trans. (Institute Ind. Eng., vol. 28, no. 1, pp. 69–85, 1996, doi: 10.1080/07408179608966253.
"""
struct Non_Uniform_Mutation <: MutationMethod
    method_name::String
    γ::Float64
    Gmax::Float64
end
Non_Uniform_Mutation(;gamma = 0.2, Gmax = 1000) = Non_Uniform_Mutation("Non Uniform mutation", gamma, Gmax)

function update_state!(method::Non_Uniform_Mutation, problem::Problem{T}, iteration::Int, state::MutationState) where {T}
    
    # select the posistion where a mutation will be occured
    positions = rand(Bernoulli(method.γ), problem.dimension)
   
    for i in findall(x->x, positions)
        r1, r2 = rand(), rand()
        fg = (iteration <= method.Gmax) ? r2 * (1 - iteration / method.Gmax) : 0
        state.x[i] = (r1 < 0.5 ) ? trunc(Int, state.x[i] + (problem.upper[i] - state.x[i]) * fg) : trunc(Int, state.x[i] - ( state.x[i] - problem.lower[i]) * fg) + 1
    end
    
    state.x, state.f_x, 1# we've invoked the similation only one time
end



