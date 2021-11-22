function default_neighbor(x::AbstractArray{T}, upbound, lowBound, objfun) where T
    neighbors=DataFrame(:x=>[], :f=>[], :edited_dim=>[])
    nbr_simulation=0
    for i in 1:length(x)
        x1=copy(x)
        x2=copy(x)
        x1[i] += 1
        x2[i] -= 1
        if isFeasible(x1,upbound, lowBound)  
            push!(neighbors,(x1,objfun(x1),i))
            nbr_simulation += 1
        end
        if isFeasible(x2,upbound, lowBound)  
            push!(neighbors,(x2,objfun(x2),i))
            nbr_simulation += 1
        end
    end
    neighbors, nbr_simulation
end


struct TabuSearch{Tn, Ttenur} <: LowLevelHeuristic
    method_name::String
    neighbor::Tn
    tenure_increase::Ttenur
    tenure_decrease::Ttenur
end
TabuSearch(;neighbor = default_neighbor,
    tenure_increase = 3,
    tenure_decrease = 4) =
    TabuSearch("Tabu Search", neighbor, tenure_increase, tenure_decrease)


mutable struct TabuSearchState{Tx,T} <: State
    x::Tx# the best
    iteration::Int
    x_current::Tx
    f_x_current::T
    f_x::T
    increase_tabu_list::Dict
    decrease_tabu_list::Dict
    nbr_simulation::Int
end

function initial_state(method::TabuSearch, problem::Problem{T}) where T
    result=problem.objective(problem.x_initial)
    # Store the best state ever visited
    best_x = copy(problem.x_initial)
    TabuSearchState(copy(best_x), 1, best_x,  result, result,Dict(),Dict(),0)
end

function update_state!(method::TabuSearch, problem::Problem{T}, iteration::Int, state::TabuSearchState) where {T}
    nbrSim = 0
    #get all the neighbors and their fitness value
    neighbors, nbr_simulation = method.neighbor(state.x_current, problem.upper, problem.lower, problem.objective)
    state.nbr_simulation += nbr_simulation
    nbrSim += nbr_simulation
    best_neighbor = neighbors[argmin(neighbors.f),:]
    #increase= true
    if best_neighbor.f >= state.f_x
        #increase= false
        while haskey(state.increase_tabu_list, best_neighbor.edited_dim) ||
                haskey(state.decrease_tabu_list, best_neighbor.edited_dim) &&
                    best_neighbor.f != Inf
            #state.tabu_list[best_neighbor.edited_dim]=state.tenure_decrease
            neighbors[argmin(neighbors.f), :f] = Inf
            best_neighbor = neighbors[argmin(neighbors.f),:]
        end
        if best_neighbor.f != Inf 
            state.x_current = best_neighbor.x
            push!(state.decrease_tabu_list,best_neighbor.edited_dim => state.iteration)
        end
    else
        state.x = best_neighbor.x
        state.f_x = best_neighbor.f
        state.x_current = best_neighbor.x
        state.f_x_current = best_neighbor.f
        state.increase_tabu_list[best_neighbor.edited_dim] = state.iteration
    end

    state.iteration += 1
    # clean up the tabu listes
    for k in keys(state.increase_tabu_list)
        if state.increase_tabu_list[k] + method.tenure_increase <= state.iteration
            delete!(state.increase_tabu_list, k)
        end
    end
    for k in keys(state.decrease_tabu_list)
        if state.decrease_tabu_list[k] + method.tenure_decrease <= state.iteration
            delete!(state.decrease_tabu_list, k)
        end
    end 

    state.x_current, state.f_x_current, nbrSim
end

function create_state_for_HH(mathod::TabuSearch, problem::Problem, HHState::HH_State)
    #best_x, result, nbrSim =  get_solution_from_archive(archive, problem, 1)
    best_x, result, nbrSim = HHState.x, HHState.x_fit, 0
    TabuSearchState(copy(best_x), 1, best_x,  result, result,Dict(),Dict(),0), nbrSim
end

    