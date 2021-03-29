#=
this is ranking and selecting procedure from Ranking and selecting comparative study

=#
using Distributions
using Combinatorics
using DataFrames
using NLsolve
include("../../../simulations/GG1K_simulation.jl")
const n0=3 #initial sampling nbr > 2
const AddsimNbr=20 #additional sampling number at each subsequence stage
const UpBound=3 #domain of each variable  is 1,2,3,4 or 5 with 3 variables so we have 245 system
const β=500 #budget of simulation (max number of simulation)
D1=collect(1:1:UpBound)
D2=collect(1:1:UpBound)
D3=collect(1:1:UpBound)


function getSamplingMeanVariance(obj::Function,X::Tuple,Z, n0::Int)
        for i in 1:n0
            push!(Z,obj(X))
        end
        μ=mean(Z)
         σ=var(Z)
        μ, σ, Z
    
end
function OCBA_Optimizer(sim::Function,D...)
    #construct systems
    X=collect(Iterators.product(D...))
    #reshap to one column
    X=reshape(X,(:,1))
    k= length(X)# size of the problem
    #perform n0 samples for each systeme
    X=convert(DataFrame,X)
    X[!,:nbrNextSim]=ones(nrow(X)).*n0
    X[!,:nbrTotalSim]=Int.(zeros(nrow(X)).*n0)
    X[!,:Y]=[[] for i in 1:nrow(X)]
    nbrSimOccure=1
    while sum(X.nbrTotalSim) < β && sum(X.nbrNextSim)>0
        #meana and variance
        transform!(X, AsTable(:).=>ByRow.(x->if x.nbrNextSim>0 
                                                 getSamplingMeanVariance(sim,x.x1,x.Y,Int(x.nbrNextSim))
                                            else x.μ, x.σ, x.Y    
                                            end).=> :tmp)
        select!(X, :x1, AsTable(:).=>ByRow.([ x->x.tmp[1], x->x.tmp[2], x->vcat(x.Y,x.tmp[3]),x->x.nbrNextSim+x.nbrTotalSim, x-> 0])
                                                .=> [:μ,:σ,:Y,:nbrTotalSim, :nbrNextSim]);
        println("Finish take samples")
        
        indice_best=argmax(X.μ)
        j=1;
        T=sum(X.nbrTotalSim)
        if j==indice_best j=2 end 
        #calculate ni for each i in
        sum1=0
        sum2=0
        σ= copy(X.σ)
        #println(σ)
        for i in 1:nrow(X)
            if i!= indice_best
                 #println("$i  $j  ",  X.σ[i]/X.σ[j] ,"  ",(X.μ[indice_best]-X.μ[j]),"  ", (X.μ[indice_best]-X.μ[i]))
                 #println("$i  $j  ",  σ[i],"   ",σ[j] ,"  ", σ[i]/σ[j])
                 sum1 += σ[i]/σ[j]*((X.μ[indice_best]-X.μ[j])/(X.μ[indice_best]-X.μ[i]))^2
                 sum2+= (1/X.σ[i])* (X.σ[i]^2/X.σ[j]^2*((X.μ[indice_best]-X.μ[j])/(X.μ[indice_best]-X.μ[i]))^4)
            end
        end
        println("Finish calculating sums")
        nl=T*1/(sum1+σ[indice_best]^0.5*sum2^0.5)
        nt=(sum1+X.σ[indice_best]^0.5*sum2^0.5)
        X.nbrNextSim[j]=floor(nl)
        #updating all the other next simulations
        for i in 1:nrow(X)
            if i != j && i != indice_best
                X.nbrNextSim[i]= floor(nl*( X.σ[i]/X.σ[j]*((X.μ[indice_best]-X.μ[j])/(X.μ[indice_best]-X.μ[i]))^2))
            end
        end
        #updating x best next iterations
        X.nbrNextSim[indice_best]= floor(nl*X.σ[indice_best]^0.5*sum2^0.5)
        #arroundi tout ces chifres                           
        transform!(X, AsTable(:).=>ByRow.(x->Int(max(0,floor(x.nbrNextSim-x.nbrTotalSim)))).=> :nbrNextSim)
        println("Finish updating nbrNextSim")
        println(sum(X.nbrTotalSim))
    end
    X
end

startTime=time()
X=OCBA_Optimizer(sim_GG1K,D1,D2,D3);

X[argmin(X.μ[:]),:]
delay=time()-startTime
first(X[:,[:μ, :x1]],27)
