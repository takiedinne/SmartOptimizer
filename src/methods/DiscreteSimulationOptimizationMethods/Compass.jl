#=
implementation from the article wroted by Discrete optimization via simulation using COMPASS
L. Jeff Hong and Barry L. Nelson
=#
using DataFrames
include("../../../simulations/GG1K_simulation.jl")

const m=5 #number of solution sampled at each iteration
const β=5
const γ=0.01
const MAX_SIMULATION=2000


function SimulationAllocationRule(k::Int)
    min(5, floor(β*log(k)^(1+γ))+1)
end
function isFeasible(x, upBound, LowBound)
    isFeasible=true
    if sum(x.>upBound)>0 || sum(x.<LowBound)>0
        isFeasible=false
    end
    isFeasible
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

function COMPASS(sim::Function, x0, upbound, lowBound)
    x_opt=x0
    k=0#current iteration
    V=DataFrame(:x=>[],:addSimulation=>Int[],:NumberSimulationDone=>Int[], :meanSampling=>[])# List of visited solutions
    addSim=SimulationAllocationRule(k)
    fit_sum=0
    for i in 1: addSim
        fit_sum+=sim(x0)
    end
    push!(V,(x0,0,addSim,fit_sum/addSim))
    #get m sampling solution from all the solution space
    k+=1
    PromosingArea=[]
    for i in 1:m
        push!(PromosingArea,rand(lowBound:upbound,length(x0)))
    end
    
    
    while sum(V.NumberSimulationDone)<MAX_SIMULATION
        #we add all the solution in promsing area to value
        addSim=SimulationAllocationRule(k)
        for i in 1:size(PromosingArea)[1]
            if PromosingArea[i] in V.x
                j=findfirst(x->x==PromosingArea[i],V.x)
                V.addSimulation[j]+=addSim
            else
                push!(V,(PromosingArea[i],addSim,0,0))
            end
        end
        #run simulations
        for i in 1:nrow(V)
            fit_sum=0
            for j in 1:V.addSimulation[i]
                fit_sum+=sim(V.x[i])
            end
            V.meanSampling[i]=(V.meanSampling[i]*V.NumberSimulationDone[i]+fit_sum)/(V.NumberSimulationDone[i]+V.addSimulation[i])
            V.NumberSimulationDone[i]+=V.addSimulation[i]
            V.addSimulation[i]=0
        end
        x_opt=V.x[argmin(V.meanSampling)]
        #here we generate the promosing area for the next iteration and we sample from there m
        #here we are in integer search space so we alter only one dimension using get neighbors
        neighbors=GetNeighbors(x_opt,upbound,lowBound)
        if length(neighbors)==1
            break
        end
        # we sample m solution from these neighbors
        indexes=rand(1:length(neighbors),m)
        
        PromosingArea=[]
        for i in 1:length(indexes)
            push!(PromosingArea,neighbors[indexes[i]])
        end
        k+=1
    end
    V.x[argmin(V.meanSampling)], V.meanSampling[argmin(V.meanSampling)]
end
t= time_ns()
V=COMPASS(sim_GG1K,ones(3),20,1)
delay= time_ns()-t 

sim_GG1K([5,7,6])