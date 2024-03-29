#= this algorithm is devlopped as described STOCHASTIC
 OPTIMIZATION WITH ESTIMATIO
WEI-BO GONGy, YU-CHI HOz, AND WENGANG ZHAI
=#
function SimulationAllocationRule(method, k::Int)
    floor(method.c * log(k + method.k_0 + 1))
end

struct StochasticComparison <:LowLevelHeuristic
    method_name::String
    simulationAllocationRule::Function # function which return nbr of simulation to perform in iteration k
    neighborhood::Function # here this function give one neigbor for the solution on input by default i will use random_x!
    c
    k_0
end

StochasticComparison(;simulationAllocationRule=SimulationAllocationRule, neighbor=random_x!, c=1, k_0=1) = 
    StochasticComparison("Stochastic Comparison", simulationAllocationRule, neighbor, c, k_0)

mutable struct StochasticComparisonState{T} <:State
    x::Array{T,1}
    x_condidate::Array{T,1}
    f_x::Real
    mean_f_condidate::Real
    nbr_simulated_optimal::Integer
    iteration::Integer
end

function initial_state(method::StochasticComparison, problem::Problem{T}) where T
    f = problem.objective(problem.x_initial)
    return StochasticComparisonState(problem.x_initial,copy(problem.x_initial), f, f, 1, 1)    
end

function update_state!(method::StochasticComparison, problem::Problem{T}, iteration::Int, state::StochasticComparisonState) where {T}
    # get a condidate solution from the neiborhood here we assume that all the search space construct the neighborhood structre
    # and also we assume that the probabilty of moving to anthor solution is equal for all the space
    # so we use uniform distribution to create the condidate solution but we must assert that the condidate is not 
    # equal to the current optimal solution
    #method.neighborhood(state.x_condidate, problem.dimension, upper = problem.upper, lower = problem.lower)
    default_neighbor!(state.x, state.x_condidate, problem.upper, problem.lower)
    # to ensure that the generating condidate solution don't equal the current solution
    nbrOfComparison = SimulationAllocationRule(method, state.iteration)
    comparisonTest = true
    nbr_simulation_performed = 0
    sum_f = 0
    sum_f_condidate = 0
    for i in 1:nbrOfComparison
        f = problem.objective(state.x)
        sum_f += f
        nbr_simulation_performed += 1
        f_condidate = problem.objective(state.x_condidate)
        sum_f_condidate += f_condidate
        if f_condidate > f
            #println("failed at iteration $iteration")
            comparisonTest = false
            break
        end
    end
    
    if comparisonTest
        #println("succeed at iteration $iteration")
        state.x= state.x_condidate
        state.f_x= sum_f_condidate / nbrOfComparison
        state.nbr_simulated_optimal = nbrOfComparison
    else
        state.f_x = (state.f_x * state.nbr_simulated_optimal + sum_f) / (state.nbr_simulated_optimal + nbr_simulation_performed)
        state.nbr_simulated_optimal += nbr_simulation_performed
    end
    state.iteration += 1
    state.x, state.f_x, nbr_simulation_performed * 2
end

function create_state_for_HH(method::StochasticComparison, problem::Problem, HHState::HH_State)
    #x, f = archive.x[argmin(archive.fit)], minimum(archive.fit)
    x, f = HHState.x, HHState.x_fit
    return StochasticComparisonState(x,copy(x), f, f, 1, 1), 0   
end

