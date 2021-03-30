
#= this algorithm is devlopped as described from
ALREFAEI, Mahmoud H. et ANDRADÓTTIR, Sigrún. 
Discrete stochastic optimization using 
variants of the stochastic ruler method. 
Naval Research Logistics (NRL), 2005, vol. 52, no 4, p. 344-360.=#
const M=3
include("../../../simulations/GG1K_simulation.jl")
function isFeasible(x, upBound, LowBound)
    isFeasible=true
    for i in 1: length(x)
        if x[i]<LowBound || x[i]>upBound
            isFeasible=false
        end
    end
    isFeasible
end
function StochasticRuler(sim::Function, x0, upBound, lowBound,dim::Int)
    f_lowBound=-200 #large number
    f_upBound=700
   
    k=0# current iterations
    X_opt=x0 #the actuelle optimal solution
    fit_opt=sim(x0)#the fitness value for the optimal solution
    currentX=x0 
    current_fit=fit_opt
    A=Dict(x0=>current_fit) #sum of sampling values for each visited x
    C=Dict(x0=>1) #number of visited times foreach x
    # the neighborhood structure is all x that differ from the current solution in dimension
    #here R is 1 for all neighbor 
    while k<2000
        dimToChange=rand(1:dim)
        changeValue= rand([-1,1])
        Z=copy(currentX)
        Z[dimToChange]+=changeValue
        if isFeasible(Z,upBound,lowBound)
            successInTest=true
            for i in 1:M
                fit_Z=sim_GG1K(Z)
                if haskey(A,Z)
                    A[Z]=(A[Z]*C[Z] + fit_Z)/(C[Z]+1)
                    C[Z]+=1
                else
                    push!(A,Z=>fit_Z)
                    push!(C,Z=>1)
                end
                uniformNumber=rand(f_lowBound:f_upBound)
                if fit_Z >uniformNumber
                    successInTest=false
                    break
                end
            end
            if successInTest
                currentX=Z
            end
            X_opt=argmin(A)
            fit_opt= A[X_opt]
            k+=1
        end
    end
    X_opt, fit_opt
end
StochasticRuler(sim_GG1K,ones(3),500,0,3)
sim_GG1K([0,4,2])