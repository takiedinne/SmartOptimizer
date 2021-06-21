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

function GetNeighbors(x, upbound, lowBound)
    dim= length(x)
    neighbors=[]
    push!(neighbors,x)
    for i in 1:dim
        x1=copy(x)
        x2=copy(x)
        x1[i]+=1
        x2[i]-=1
        if isFeasible(x1,upbound, lowBound)  
            push!(neighbors,x1)
        end
        if isFeasible(x2,upbound, lowBound)  
            push!(neighbors,x2)
        end
    end
    neighbors
end

function SimulationAllocationRule(method::COMPASS_Searcher, k::Int)
    min(5, floor(method.β*log(k)^(1+method.γ))+1)
end

const COMPASSDefaultPrametres = Dict( :neighborsSearcher => GetNeighbors,
                                      :SimulationAllocationRule => SimulationAllocationRule,
                                      :m => 5,
                                      :β => 5,
                                      :γ => 0.01)

COMPASS_Searcher(;neighborsSearcher = COMPASSDefaultPrametres[:neighborsSearcher],
                    SAR = COMPASSDefaultPrametres[:SimulationAllocationRule],
                    m= COMPASSDefaultPrametres[:m],
                    β = COMPASSDefaultPrametres[:β],
                    γ = COMPASSDefaultPrametres[:γ])= COMPASS_Searcher("COMPASS", neighborsSearcher, SAR, m, β, γ)


mutable  struct COMPASS_SearcherState{T} <:State
    x_opt::AbstractArray{T,1} # here x_opt is the same x_current
    f_opt
    k::Int #iteration counter
    V::DataFrame #historic information (nbr of sim done, mean of sampling ...etc)
    PromosingArea::AbstractArray{Any,1}# m neighbors for the current x
end 

function initial_state(method::COMPASS_Searcher, problem::Problem{T}) where {T<:Number}
    lower= problem.lower
    upper = problem.upper
    objfun = problem.objective
    initial_x = problem.x_initial
    dimension = problem.dimension
    V=DataFrame(:x=>[],:addSimulation=>Int[],
                    :NumberSimulationDone=>Int[], :meanSampling=>[])# List of visited solutions
    
    addSim = method.SimulationAllocationRule(method,0)
    fit_sum=0
    for i in 1: addSim
        fit_sum+=objfun(initial_x)
    end
    push!(V,(initial_x,0,addSim,fit_sum/addSim))
    PromosingArea=[]
    # here we initialise the promosing area by random points from the search space 
    for i in 1:method.m
        x=copy(initial_x)
        random_x!(x,dimension, upper=upper, lower=lower)
        push!(PromosingArea,x)
    end
    COMPASS_SearcherState(initial_x, V[1,:].meanSampling, 0, V, PromosingArea)
end

function update_state!(method::COMPASS_Searcher, problem::Problem{T}, iteration::Int, state::COMPASS_SearcherState) where {T}
    #we add all the solution in promsing area to value
    addSim=SimulationAllocationRule(method, state.k)
    PromosingArea=state.PromosingArea
    V=state.V
    f= problem.objective
    for i in 1:size(PromosingArea)[1]
        if PromosingArea[i] in V.x
            j=findfirst(x->x==PromosingArea[i],V.x)
            V.addSimulation[j]+=addSim
        else
            push!(V,(PromosingArea[i],addSim,0,0))
        end
    end
    nbrSim=0 #nbr of simulations counter
    #run simulations
    for i in 1:nrow(V)
        fit_sum=0
        for j in 1:V.addSimulation[i]
            fit_sum+=f(V.x[i])
            
        end
        V.meanSampling[i]=(V.meanSampling[i]*V.NumberSimulationDone[i]+fit_sum)/(V.NumberSimulationDone[i]+V.addSimulation[i])
        V.NumberSimulationDone[i]+=V.addSimulation[i]
        nbrSim+=V.addSimulation[i]
        V.addSimulation[i]=0
    end
    state.x_opt=V.x[argmin(V.meanSampling)]
    #here we generate the promosing area for the next iteration and we sample from there m
    #here we are in integer search space so we alter only one dimension using get neighbors
    neighbors=method.neighborsSearcher(state.x_opt,problem.upper,problem.lower)
    
    # we sample m solution from these neighbors
    indexes=rand(1:length(neighbors),method.m)
    
    PromosingArea=[]
    for i in 1:length(indexes)
        push!(PromosingArea,neighbors[indexes[i]])
    end
    state.PromosingArea=PromosingArea
    state.k+=1
    V.x[argmin(V.meanSampling)], V.meanSampling[argmin(V.meanSampling)], nbrSim
end

function isFeasible(x, upBound, LowBound)
    return !(sum(x.>upBound)>0 || sum(x.<LowBound)>0)
end

function create_state_for_HH(method::COMPASS_Searcher, problem::Problem{T}, archive) where {T<:Number}
    xBestArchive = archive.x[argmin(archive.fit)]
    V=DataFrame(:x=>[],:addSimulation=>Int[],
                    :NumberSimulationDone=>Int[], :meanSampling=>[])# visited solutions List
    
    
    push!(V,(xBestArchive,0,1,minimum(archive.fit)))
    PromosingArea=[]
    # here we initialise the promosing area 
    neighbors=method.neighborsSearcher(xBestArchive,problem.upper,problem.lower)
    # we sample m solution from these neighbors
    indexes=rand(1:length(neighbors),method.m)
    
    PromosingArea=[]
    for i in 1:length(indexes)
        push!(PromosingArea,neighbors[indexes[i]])
    end
    COMPASS_SearcherState(xBestArchive, V[1,:].meanSampling, 0, V, PromosingArea), 0
end
