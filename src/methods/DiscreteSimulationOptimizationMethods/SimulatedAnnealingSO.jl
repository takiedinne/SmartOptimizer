"""
this algorithm is implemented like the way presented in [1]
[1]: Prudius, A. A., & Andradóttir, S. (2012). Averaging frameworks for
simulation optimization with applications to simulated annealing. Naval Research Logistics, 59(6),
411–429. https://doi.org/10.1002/nav.21496
"""
cooling_temperature(n, C) = C/log(n+10) 
function N_L(x::AbstractArray{T}, upper, lower) where T
    x_proposal = copy(x)
    for i in eachindex(x)
        tmp = x[i] + rand([-1,0,1])
        if lower[i] < tmp < upper[i] # workaround because all types might not have randn
            x_proposal[i] =  tmp
        else
            x_proposal[i] = x[i]
        end
    end
    return x_proposal
end
function N_G(x::AbstractArray{T}, upper, lower) where T
    x_proposal = copy(x)
    random_x!(x_proposal, length(x_proposal), upper = upper, lower = lower)
    return x_proposal
end


struct SimulatedAnnealingSO{Tn, Ttemp} <:LowLevelHeuristic
    method_name::String
    neighbor::Tn
    temperature::Ttemp
    coolingRate
    k::Int # nbr of observation each time
    keep_best::Bool 
end

SimulatedAnnealingSO(;neighbor = N_L,
                    temperature = cooling_temperature,
                    coolingRate = 20,
                    k = 10,
                    keep_best::Bool = true) =
      SimulatedAnnealingSO("Simulated Annealing SO adapted",neighbor, temperature, coolingRate, k, keep_best)

mutable struct SimulatedAnnealingSOState{Tx,T} <:State
    x::Tx# the best
    iteration::Int
    x_current::Tx #the best 
    x_proposal::Tx
    f_x_current::T
    f_proposal::T
    f_x::T #
    # we keep the trace of the visited solution and their nbr of simulations
    # we use DataFrame structure
    visitedSolutions::DataFrame # contains two column the solution and the 
                                # mean and the nbr of replication
end

function initial_state(method::SimulatedAnnealingSO, problem::Problem{T}) where T
    #step 0 
    result = 0
    for i in 1: method.k
        result += problem.objective(problem.x_initial)
    end
    result /= method.k
    # Store the best state ever visited
    best_x = copy(problem.x_initial)
    visitedSolutions = DataFrame(x= [copy(best_x)], fit = [result], observationsNbr = [method.k])

    SimulatedAnnealingSOState(copy(best_x), 1, best_x, copy(best_x), result, result,result, visitedSolutions)
end

function update_state!(method::SimulatedAnnealingSO, problem::Problem{T}, iteration::Int, state::SimulatedAnnealingSOState) where {T}
    nbrSim = 0
    #step 1
    # Randomly generate a neighbor of our current state (uniform distribution)
    state.x_proposal = method.neighbor(state.x_current, problem.upper, problem.lower)
    
    #step 2
    # Evaluate the cost function at the proposed state
    avg_fit_proposal = 0
    avg_fit_current = 0
    for i in 1:method.k
        avg_fit_proposal += problem.objective(state.x_proposal)
        avg_fit_current += problem.objective(state.x_current)
    end
    nbrSim += 2 * method.k

    state.f_x_current = avg_fit_current / method.k
    state.f_proposal = avg_fit_proposal / method.k

    #check if we've already visited this proposal solution
    visitedSolutions = state.visitedSolutions
    if state.x_proposal in visitedSolutions.x
        index_proposal = findfirst(x-> x == state.x_proposal,visitedSolutions.x)
        visitedSolutions.fit[index_proposal] = (visitedSolutions.fit[index_proposal] * visitedSolutions.observationsNbr[index_proposal] + avg_fit_proposal) / (visitedSolutions.observationsNbr[index_proposal] + method.k)
        visitedSolutions.observationsNbr[index_proposal] += method.k
    else
        push!(visitedSolutions, (state.x_proposal, avg_fit_proposal/method.k, method.k))
    end
    # update the information about the current solution
    index_current = findfirst(x-> x == state.x_current,visitedSolutions.x)
    visitedSolutions.fit[index_current] = (visitedSolutions.fit[index_current] * visitedSolutions.observationsNbr[index_current] + avg_fit_current) / (visitedSolutions.observationsNbr[index_current] + method.k)
    visitedSolutions.observationsNbr[index_current] += method.k

    #step 3
    t = method.temperature(state.iteration, method.coolingRate)
    # If proposal is inferior, we move to it with probability p
    p = exp(-max((state.f_x_current - state.f_proposal),0) / t)
    if rand() <= p
        copyto!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal
    end
    #step 4
    state.iteration += 1
    optIndex = argmin(state.visitedSolutions.fit)
    state.x = state.visitedSolutions.x[optIndex]
    state.f_x = state.visitedSolutions.fit[optIndex]
    
    state.x_current, state.f_x_current, nbrSim
end

function create_state_for_HH(method::SimulatedAnnealingSO, problem::Problem, HHState::HH_State)
    result=HHState.x_fit
    # Store the best state ever visited
    best_x = HHState.x
    SimulatedAnnealingState(copy(best_x), 1, best_x, copy(best_x), result, result,result), 0
end
