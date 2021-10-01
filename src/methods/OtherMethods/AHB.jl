# implmentation as it is described in Handbook of simulation optimization page 38(45)

using DataFrames
include("../../../simulations/GG1K_simulation.jl")

const t=5 #number of solution sampled at each iteration
const β=5
const γ=0.01
const MAX_SIMULATION=2000


function SimulationAllocationRule(k::Int)
    min(5, floor(β*log(k)^(1+γ))+1)
end

function GetNeighborsBoundaries(x, V,upBound, lowBound)
    dim= length(x)
    neighbors=[]
    L , U = [], []
    values=[]
    for i in 1:dim
        a=[]
        for j in 1:length(V)
            push!(a,V[j][i])
        end
        push!(values,a)
    end
    #get boundaries
    for i in 1:dim
        opt_value=x[i]
        tmp=values[i]
        a= tmp[tmp.<opt_value]
        b= tmp[tmp.>opt_value]
        if length(a)>0
            push!(L,max(a...))
        else push!(L,lowBound) end
        if length(b)>0
            push!(U,min(b...))
        else push!(U,upBound) end 
    end
    U,L

end

function AdaptiveHyperBox(sim::Function, x0, upbound, lowBound)
    x_opt=x0
    m=0#current iteration
    V=DataFrame(:x=>[],:NumberSimulationDone=>Int[], :meanSampling=>[])# List of visited solutions
    addSim=SimulationAllocationRule(m)
    fit_sum=0
    for i in 1: addSim
        fit_sum+=sim(x0)
    end
    push!(V,(x0,addSim,fit_sum/addSim))
    #get m sampling solution from all the solution space
    m+=1
    PromosingArea=[]
    for i in 1:t
        push!(PromosingArea,rand(lowBound:upbound,length(x0)))
    end
    push!(PromosingArea,x_opt)
    unique!(PromosingArea)
    up=[]# upBound for each dimension
    low=[]# lowBound for each dimension

    while sum(V.NumberSimulationDone)<MAX_SIMULATION
        #we add all the solution in promsing area to value
        addSim=SimulationAllocationRule(m)
        for i in 1:size(PromosingArea)[1]
            fit_sum=0
            for j in 1: addSim
                fit_sum+=sim(PromosingArea[i])
            end
            #we update the mean and the nbr of simylation donne
            if PromosingArea[i] in V.x
                indice=findfirst(x->x==PromosingArea[i],V.x)
                V.meanSampling[indice]=(V.meanSampling[indice]* V.NumberSimulationDone[indice]
                                        + fit_sum)/( V.NumberSimulationDone[indice]+addSim)
                V.NumberSimulationDone[indice]+=addSim
            else
                push!(V,(PromosingArea[i],addSim,fit_sum/addSim))
            end
        end
        
        x_opt=V.x[argmin(V.meanSampling)]
        #here we generate the promosing area for the next iteration and we sample from there m
        #here we are in integer search space so we alter only one dimension using get neighbors
        neighborsUpBound , neighborsLowBound =GetNeighborsBoundaries(x_opt,V.x,upbound,lowBound)
        
        PromosingArea=[]
        tmpData=rand(neighborsLowBound[1]:neighborsUpBound[1],t)
        for i in 2:length(x_opt)
            tmpData= hcat(tmpData,rand(neighborsLowBound[i]:neighborsUpBound[i],t))
        end
        for i in 1:t
            push!(PromosingArea,tmpData[i,:])
        end
        m+=1
    end
    #V.x[argmin(V.meanSampling)], V.meanSampling[argmin(V.meanSampling)]
    V
end

V=AdaptiveHyperBox(sim_GG1K,ones(3),5,1)

nrow(V)
V[argmin(V.meanSampling),:]