"""
To do put here the references 
"""
abstract type LocalSearch <: LowLevelHeuristic end

mutable struct LocalSearchState{T} <: State
    x::AbstractArray{T,1} # the best solution here it's the child
    f_x # the fitness of the new solution
    currentDepth:: Integer
end 

function create_state_for_HH(method::LocalSearch, problem::Problem{T}, HHState::HH_State) where {T<:Number}
    # get two solution to be parents from the archive
    x, fit = HHState.x, HHState.x_fit
    LocalSearchState(x, fit, 0), 0
end

"""
we have implemented the steepest descent (Hill climbing) local search as it is described in [1].
the steepest desent method is a local search where we chose to move to the best solution
in the neighborhood. so according the way that we are generating the neighborhood we can have
many variant
--> to do search many variant of neighber generator

[1]: Crainic, T. G., & Toulouse, M. (2006). Parallel Strategies for Meta-Heuristics.
 Handbook of Metaheuristics, 475–513. https://doi.org/10.1007/0-306-48056-5_17
"""

# default neighbors are those who are different from the original solution in one dimension
function defaultNeighborsGenerator(x, upbound, lowBound)  
    dim = length(x)
    neighbors=[[i == j ? x[i] + 1 : x[i] for i in 1:dim ] for j in 1:dim]
    append!(neighbors, [[i == j ? x[i] - 1 : x[i] for i in 1:dim ] for j in 1:dim])
    #neighbors
    return neighbors[[isFeasible(neighbors[i], upbound,lowBound) for i in 1:length(neighbors)]]
end

struct SteepestDescentMethod <: LocalSearch
    method_name::String
    maxSearchDepth #nbr max of iterations # to do find a good value for this parameter
    neighborGenerator
end
SteepestDescentMethod(;depth = 5, neighborGenerator = defaultNeighborsGenerator) = 
    SteepestDescentMethod("Steepest Descent Method", depth, neighborGenerator)

function update_state!(method::SteepestDescentMethod, problem::Problem{T}, iteration::Int, state::LocalSearchState) where {T}
    nbrOfsim = 0
    while state.currentDepth <= method.maxSearchDepth
        #get all feasible solution in the neighborhood
        neighbors = method.neighborGenerator(state.x, problem.upper, problem.lower)
        best_local_fit = state.f_x
        best_x = state.x
        for n in neighbors
            f_n = problem.objective(n)
            nbrOfsim += 1
            if f_n >= best_local_fit
                best_x = n
                best_local_fit = f_n
            end
        end
        if best_x == state.x
            # there is no improvement
            break
        end
        state.currentDepth += 1
        state.x = best_x
        state.f_x = best_local_fit
    end
    state.x, state.f_x, nbrOfsim # we've invoked the similation only one time
end

""" 
the first improvement or greedy ascent is a local search where we move to the enxt solution
in the neighberhood if it is better than the current solution and we don't need to evaluate all
the solutions. it is implemented as the way described in [2]
[2]: Ochoa, G., Verel, S., & Tomassini, M. (2010). First-improvement vs. best-improvement 
local optima networks of NK landscapes. Lecture Notes in Computer Science 6238 LNCS(PART 1), 104–113.
"""

struct FirstImprovementMethod <: LocalSearch
    method_name::String
    maxSearchDepth #nbr max of iterations # to do find a good value for this parameter
    neighborGenerator
end
FirstImprovementMethod(;depth = 5, neighborGenerator = defaultNeighborsGenerator) = 
    FirstImprovementMethod("First Improvement Method", depth, neighborGenerator)

function update_state!(method::FirstImprovementMethod, problem::Problem{T}, iteration::Int, state::LocalSearchState) where {T}
    nbrOfsim = 0
    while state.currentDepth <= method.maxSearchDepth
        #get all feasible solution in the neighborhood
        neighbors = shuffle(method.neighborGenerator(state.x, problem.upper, problem.lower))
        best_local_fit = state.f_x
        best_x = state.x
        for n in neighbors
            f_n = problem.objective(n)
            nbrOfsim += 1
            if f_n >= best_local_fit
                best_x = n
                best_local_fit = f_n
                break # if there is an improvement we stop searching in the neighborhood
            end
        end
        if best_x == state.x
            # there is no improvement
            break
        end
        state.currentDepth += 1
        state.x = best_x
        state.f_x = best_local_fit
    end
    state.x, state.f_x, nbrOfsim # we've invoked the similation only one time
end
