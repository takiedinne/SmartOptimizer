using DataFrames
include("../../../simulations/GG1K_simulation.jl")


function default_neighbor!(x::AbstractArray{T}, bounds::Tuple{T,T}, objfun) where T
    neighbors=DataFrame(:x=>[], :f=>[], :edited_dim=>[])
    nbr_simulation=0
    for i in 1:length(x)
        x1=copy(x)
        x2=copy(x)
        if x1[i]+1 <= bounds[2]
            x1[i]+=1
            push!(neighbors,(x1,objfun(x1),i))
            nbr_simulation+=1
        end
        if x2[i]-1 >= bounds[1]
            x2[i]-=1
            push!(neighbors,(x2,objfun(x2),i))
            nbr_simulation+=1
        end
    end
    neighbors, nbr_simulation
end


struct TabuSearch{Tn, Ttenur} 
    neighbor::Tn
    tenure_increase::Ttenur
    tenure_decrease::Ttenur
    
end
TabuSearch(;neighbor = default_neighbor!,
    tenure_increase = 3,
    tenure_decrease = 4) =
    TabuSearch(neighbor, tenure_increase, tenure_decrease)


mutable struct TabuSearchState{Tx,T} 
    x::Tx# the best
    iteration::Int
    x_current::Tx
    f_x_current::T
    f_x::T
    increase_tabu_list::Dict
    decrease_tabu_list::Dict
    nbr_simulation::Int
end

function initial_state(method::TabuSearch, options, objfun, initial_x::AbstractArray{T}) where T
    result=objfun(initial_x)
    # Store the best state ever visited
    best_x = copy(initial_x)
    TabuSearchState(copy(best_x), 1, best_x,  result, result,Dict(),Dict(),0)
end

function random_initial_point(dim, bounds)
    x_initial=rand(bounds[1]:bounds[2],dim)
    x_initial
end


function TabuSearchAlgo(objfun, bounds, dim) 
    method= TabuSearch()
    state=initial_state(method, [],objfun,random_initial_point(dim,bounds))
    while state.iteration<1000
        println(state.x_current)
        println(state.x)
        println("******************************")
        #get all the neighbors and their fitness value
        neighbors, nbr_simulation=method.neighbor(state.x_current, bounds, objfun)
        state.nbr_simulation+=nbr_simulation
        best_neighbor=neighbors[argmin(neighbors.f),:]
        #increase= true
        if best_neighbor.f >= state.f_x
            #increase= false
            while (haskey(state.increase_tabu_list, best_neighbor.edited_dim) ||
                    haskey(state.decrease_tabu_list, best_neighbor.edited_dim)) &&
                     best_neighbor.f != Inf
                #state.tabu_list[best_neighbor.edited_dim]=state.tenure_decrease
                neighbors[argmin(neighbors.f), :f]=Inf
                best_neighbor=neighbors[argmin(neighbors.f),:]
            end
            if best_neighbor.f != Inf 
                state.x_current=best_neighbor.x
                push!(state.decrease_tabu_list,best_neighbor.edited_dim=>state.iteration)
            end
        else
            state.x=best_neighbor.x
            state.f_x=best_neighbor.f
            state.x_current=best_neighbor.x
            state.f_x_current=best_neighbor.f
            state.increase_tabu_list[best_neighbor.edited_dim]=state.iteration
        end

        state.iteration += 1
        # clean up the tabu listes
        for k in keys(state.increase_tabu_list)
            if state.increase_tabu_list[k]+method.tenure_increase <= state.iteration
                delete!(state.increase_tabu_list, k)
            end
        end
        for k in keys(state.decrease_tabu_list)
            if state.decrease_tabu_list[k]+method.tenure_decrease <= state.iteration
                delete!(state.decrease_tabu_list, k)
            end
        end

    end    
    state 
end

result = TabuSearchAlgo(sim_GG1K,(1,5),3)
result.x
result.nbr_simulation
result.f_x
result.decrease_tabu_list