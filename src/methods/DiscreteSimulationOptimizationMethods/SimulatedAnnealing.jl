include("../../../simulations/GG1K_simulation.jl")

log_temperature(t) = 1 / log(t)

constant_temperature(t) = 1.0

function default_neighbor!(x::AbstractArray{T}, x_proposal::AbstractArray, bounds::Tuple{T,T}) where T
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        tmp = x[i] + T(round(randn()))
        if bounds[1]<tmp< bounds[2] # workaround because all types might not have randn
            x_proposal[i] = x[i] + tmp
        else
            x_proposal[i] = x[i]
        end
    end
end

struct SimulatedAnnealing{Tn, Ttemp} 
    neighbor::Tn
    temperature::Ttemp
    keep_best::Bool # not used!?
    
end
SimulatedAnnealing(;neighbor = default_neighbor!,
                    temperature = log_temperature,
                    keep_best::Bool = true) =
      SimulatedAnnealing(neighbor, temperature, keep_best)
s= SimulatedAnnealing()

mutable struct SimulatedAnnealingState{Tx,T} 
    x::Tx# the best
    iteration::Int
    x_current::Tx
    x_proposal::Tx
    f_x_current::T
    f_proposal::T
    f_x::T
end

function initial_state(method::SimulatedAnnealing, options, objfun, initial_x::AbstractArray{T}) where T

    result=objfun(initial_x)
    # Store the best state ever visited
    best_x = copy(initial_x)
    SimulatedAnnealingState(copy(best_x), 1, best_x, copy(initial_x), result, result,result)
end
function random_initial_point(dim, bounds)
    x_initial=rand(bounds[1]:bounds[1],dim)
    x_initial
end


function SimulatedAnnealingAlgo(objfun, bounds, dim) 
    method= SimulatedAnnealing()
    state=initial_state(method, [],objfun,random_initial_point(dim,bounds))
    while state.iteration<2000
        # Determine the temperature for current iteration
        t = method.temperature(state.iteration)

        # Randomly generate a neighbor of our current state
        method.neighbor(state.x_current, state.x_proposal, bounds)

        # Evaluate the cost function at the proposed state
        state.f_proposal = objfun(state.x_proposal)

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
    end    
    state.x, state.f_x
end

result = SimulatedAnnealingAlgo(sim_GG1K,(1,5),3)
sim_GG1K([7,7,9])

