using Plots
#= this algorithm is devlopped as described STOCHASTIC
COMPARISON ALGORITHM FOR DISCRETE OPTIMIZATION WITH ESTIMATIO
WEI-BO GONGy, YU-CHI HOz, AND WENGANG ZHAI
=#

include("../../../simulations/GG1K_simulation.jl")
const c=1
const k_0=0

function SimulationAllocationRule(k::Int)
    floor(c*log(k+k_0+1))
end

function StochasticComparison(sim::Function, x0, upBound, lowBound)

    k=1# current iterations
    x_opt=x0 #the actuelle optimal solution
    dim=length(x0)
    nbrOfSimulation=0
    while k<2000
        # get a condidate solution from the neiborhood here we assume that all the search space construct the Naval
        # and also we assume that the probabilty of moving to anthor solution is equal for all the space
        # so we use uniform distribution to create the condidate solution but we must assert that the condidate is not 
        # equal to the current optimal solution
        x_condidate=rand(lowBound:upBound,dim)
        println(x_condidate)
        while x_condidate==x_opt
            x_condidate=rand(lowBound:upBound,dim)
        end
        nbrOfComparison=SimulationAllocationRule(k)
        comparisonTest=true
        for i in 1:nbrOfComparison
            if sim(x_condidate)> sim(x_opt)
                comparisonTest=false
                break
            end
            nbrOfSimulation+=2
        end
        if comparisonTest
            x_opt=x_condidate
        end
        k+=1
    end
    x_opt
end
StochasticComparison(sim_GG1K,ones(3),20,1)
sim_GG1K([17,8,9])