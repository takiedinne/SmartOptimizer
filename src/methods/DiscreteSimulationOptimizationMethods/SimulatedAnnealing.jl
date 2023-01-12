log_temperature(t) = 1 / log(t)

constant_temperature(t) = 1.0

function default_neighbor!(x::AbstractArray{T}, x_proposal::AbstractArray, upper, lower) where T
    @assert size(x) == size(x_proposal)
    for i in eachindex(x)
        tmp = x[i] + round(randn())
        if lower[i] < tmp < upper[i] # workaround because all types might not have randn
            x_proposal[i] =  tmp
        else
            x_proposal[i] = x[i]
        end
    end
end

struct SimulatedAnnealing{Tn, Ttemp} <:LowLevelHeuristic
    method_name::String
    neighbor::Tn
    temperature::Ttemp
    keep_best::Bool # not used!?
end

SimulatedAnnealing(;neighbor = default_neighbor!,
                    temperature = log_temperature,
                    keep_best::Bool = true) =
      SimulatedAnnealing("Simulated Annealing",neighbor, temperature, keep_best)

mutable struct SimulatedAnnealingState{Tx,T} <:State
    x::Tx # the best
    iteration::Int
    x_current::Tx
    x_proposal::Tx
    f_x_current::T
    f_proposal::T
    f_x::T
end

function initial_state(method::SimulatedAnnealing, problem::Problem{T}) where T
    result=problem.objective(problem.x_initial)
    # Store the best state ever visited
    best_x = copy(problem.x_initial)
    SimulatedAnnealingState(copy(best_x), 1, best_x, copy(best_x), result, result,result)
end

function update_state!(method::SimulatedAnnealing, problem::Problem{T}, iteration::Int, state::SimulatedAnnealingState) where {T}
    nbrSim = 0
    # Determine the temperature for current iteration
    t = method.temperature(state.iteration)

    # Randomly generate a neighbor of our current state
    method.neighbor(state.x_current, state.x_proposal, problem.upper, problem.lower)
    
    # Evaluate the cost function at the proposed state
    state.f_proposal = problem.objective(state.x_proposal)
    nbrSim +=1
    if state.f_proposal <= state.f_x_current
        # If proposal is superior, we always move to it
        copyto!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal

        # If the new state is the best state yet, keep a record of it
        if state.f_proposal < state.f_x
            state.f_x = state.f_proposal
            copyto!(state.x, state.x_proposal)
        end
    else
        # If proposal is inferior, we move to it with probability p
        p = exp(-(state.f_proposal - state.f_x_current) / t)
        if rand() <= p
            copyto!(state.x_current, state.x_proposal)
            state.f_x_current = state.f_proposal
        end
    end
    state.iteration += 1

    state.x_current, state.f_x_current, nbrSim
end

function create_state_for_HH(method::SimulatedAnnealing, problem::Problem, HHState::HH_State)
    result=HHState.x_fit
    # Store the best state ever visited
    best_x = HHState.x
    SimulatedAnnealingState(copy(best_x), 1, best_x, copy(best_x), result, result,result), 0
end
