#Important Note this method converges rapeadly 
#maybe i must add a perturbative step to avoid the local optimal

#=
implementation from the article wroted by Discrete optimization via simulation using COMPASS
L. Jeff Hong and Barry L. Nelson
=#
struct COMPASS_Searcher <: LowLevelHeuristic
    method_name::String
    neighborsSearcher
    SimulationAllocationRule
    m::Int #nbr of considred neighbors in each iteration
    #parameters for simulation allocation rule
    β::Int
    γ
end
function isFeasible(x, upBound, LowBound)
    return !(sum(x.>upBound) > 0 || sum(x.< LowBound) > 0)
end

function GetNeighbors(x, upbound, lowBound)
    dim= length(x)
    neighbors=[]
    push!(neighbors,x)
    for i in 1:dim
        x1 = copy(x)
        x2 = copy(x)
        x1[i] += 1
        x2[i] -= 1
        if !(sum(x1 .> upbound) > 0 || sum(x1 .< lowBound) > 0)  
            push!(neighbors,x1)
        end
        if !(sum(x2 .> upbound) > 0 || sum(x2 .< lowBound)>0) 
            push!(neighbors,x2)
        end
    end
    neighbors
end

function equale_simulation_allocation_rule(method::COMPASS_Searcher, k::Int)
    # equale Simulation allocation rule
    min(5, floor(method.β*log(k)^(1+method.γ))+1)
end

function vieira_simulation_allocation_rule(visited_solution, new_solution;λ=10, n_0 = 3)
    # this simulation allocation Rule from paper [1]
    # [1] J. Hélcio Vieira, K. H. Kienitz, and M. C. N. Belderrain,
    # “DISCRETE-VALUED, STOCHASTIC-CONSTRAINED SIMULATION OPTIMIZATION WITH COMPASS,” 
    #Proc. 2011 Winter Simul. Conf., pp. 3605–3616, 2011.
    
    # to implement

end

const COMPASSDefaultPrametres = Dict( :neighborsSearcher => GetNeighbors,
                                      :SimulationAllocationRule => equale_simulation_allocation_rule,
                                      :m => 5,
                                      :β => 5,
                                      :γ => 0.01)

COMPASS_Searcher(;neighborsSearcher = COMPASSDefaultPrametres[:neighborsSearcher],
                    SAR = COMPASSDefaultPrametres[:SimulationAllocationRule],
                    m= COMPASSDefaultPrametres[:m],
                    β = COMPASSDefaultPrametres[:β],
                    γ = COMPASSDefaultPrametres[:γ])= COMPASS_Searcher("COMPASS", neighborsSearcher, SAR, m, β, γ)


mutable  struct COMPASS_SearcherState{T} <:State
    x::AbstractArray{T,1} # here x is the same x_current
    f_x
    k::Int #iteration counter
    V::DataFrame #historic information (nbr of sim done, mean of sampling ...etc)
    PromisingArea::AbstractArray{Any,1}# m neighbors for the current x
end 

function initial_state(method::COMPASS_Searcher, problem::Problem{T}) where {T<:Number}
    lower = problem.lower
    upper = problem.upper
    objfun = problem.objective
    initial_x = problem.x_initial
    dimension = problem.dimension
    V = DataFrame(:x=>[],:addSimulation=>Int[],
                    :NumberSimulationDone=>Int[], :meanSampling=>[])# List of visited solutions
    
    addSim = method.SimulationAllocationRule(method,1)
    fit_sum = 0
    for i in 1:addSim
        fit_sum += objfun(initial_x)
    end
    push!(V,(initial_x, 0, addSim, fit_sum/addSim))
    PromisingArea=[] # always the size of the promising Area is equal to m
    # here we initialise the promosing area by random points from the search space 
    for i in 1:method.m
        x = copy(initial_x)
        random_x!(x, dimension, upper = upper, lower = lower)
        push!(PromisingArea,x)
    end
    COMPASS_SearcherState(initial_x, V[1,:].meanSampling, 1, V, PromisingArea)
end

function update_state!(method::COMPASS_Searcher, problem::Problem{T}, iteration::Int, state::COMPASS_SearcherState) where {T}
    #we add all the solution in promsing area to value
    addSim = method.SimulationAllocationRule(method, state.k) #adaptive number of simulation for the next iteration
    PromisingArea = state.PromisingArea # list of solution to visite 
    
    f = problem.objective #the objective function
    for i in 1:size(PromisingArea)[1]
        if PromisingArea[i] in state.V.x
            j = findfirst(x-> x == PromisingArea[i],state.V.x)
            state.V.addSimulation[j] += addSim
        else
            push!(state.V,(PromisingArea[i],addSim,0,0))
        end
    end

    nbrSim=0 #nbr of simulations counter
    #run simulations
    for i in 1:nrow(state.V)
        fit_sum = 0
        for j in 1:state.V.addSimulation[i]
            fit_sum += f(state.V.x[i])
        end
        #update the the estimated fitness value for the solutions in promosing area 
        state.V.meanSampling[i] = (state.V.meanSampling[i] * state.V.NumberSimulationDone[i] + fit_sum) / (state.V.NumberSimulationDone[i] + state.V.addSimulation[i])
        state.V.NumberSimulationDone[i] += state.V.addSimulation[i]
        nbrSim += state.V.addSimulation[i]
        state.V.addSimulation[i] = 0
    end
    state.x = state.V.x[argmin(state.V.meanSampling)]
    state.f_x = minimum(state.V.meanSampling)
    #here we generate the promising area for the next iteration and we sample from there m
    #here we are in integer search space so we alter only one dimension using get neighbors
    neighbors = method.neighborsSearcher(state.x,problem.upper,problem.lower)
    # we sample m solution from these neighbors
    indexes=rand(1:length(neighbors),method.m)
    
    PromisingArea=[]
    for i in eachindex(indexes)
        push!(PromisingArea, neighbors[indexes[i]])
    end
    state.PromisingArea = PromisingArea
    
    state.k += 1
    state.x, state.f_x, nbrSim
end

function create_state_for_HH(method::COMPASS_Searcher, problem::Problem{T}, HHState::HH_State) where {T<:Number}
    #xBestArchive = archive.x[argmin(archive.fit)]
    nbrSim = 0
    x = HHState.x
    fit =HHState.x_fit
    V=DataFrame(:x=>[x],:addSimulation=>[0],
                    :NumberSimulationDone=>[1], :meanSampling=>[fit])# visited solutions List
    
    PromisingArea = []
    # here we initialise the promosing area 
    neighbors = method.neighborsSearcher(x,problem.upper,problem.lower)
    # we sample m solution from these neighbors
    indexes=rand(1:length(neighbors), method.m)
    for i in eachindex(indexes)
        push!(PromisingArea,neighbors[indexes[i]])
    end
    COMPASS_SearcherState(x, V[1,:].meanSampling, 0, V, PromisingArea), nbrSim
end
