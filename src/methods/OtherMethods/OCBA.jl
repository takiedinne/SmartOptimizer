#=
this is ranking and selecting procedure from Ranking and selecting comparative study
this method tries to iteratively allocate nbr of simulation runs to each solution in the search space 
it explores all the search space so it is mostely used when we have finit search space

=#
struct OCBA <: LowLevelHeuristic
     method_name::String
     n0 #initial sampling nbr > 2
     AddsimNbr #additional sampling number at each subsequence stage
end
OCBA(;n0=3, AddsimNbr=20 ) = OCBA("Optimal Computing Budget Allocation", n0, AddsimNbr)
mutable struct OCBAState{T} <: State
    X::DataFrame # DataFrame search space
    x_best::Array{T,1}
    f_best::Real
end

function initial_state(method::OCBA, problem::Problem{T}) where {T<:Number}
   #construct all the search space from the upper and lower
    @assert length(problem.lower) == length(problem.upper) "lower and upper must be of same length..."
    @assert length(problem.lower) > 0 " the problem must be bounded"
    X=[]
    k= problem.dimension
    n0= method.n0
    for i in 1:k
        push!(X, collect(problem.lower[i]:1:problem.upper[i]))
    end
    X=collect(Iterators.product(X...))
    X=reshape(X,(:,1))
    #transform search space to dataframe
    X=convert(DataFrame,X)
    X[!,:nbrNextSim]=ones(nrow(X)).*n0 # each systeme would be simulated n0 times
    X[!,:nbrTotalSim]=Int.(zeros(nrow(X)).*n0) # nbr of runs performed for each system
    X[!,:Y]=[[] for i in 1:nrow(X)] # sampling mean variance for each system

    x_best=copy(problem.x_initial)
    f_best= Inf

    OCBAState(X, x_best, f_best)
end

function getSamplingMeanVariance(obj::Function,X::Tuple,Z, n0::Int)
        for i in 1:n0
            push!(Z,obj(X))
        end
        μ=mean(Z)
         σ=var(Z)
        μ, σ, Z
    
end
function update_state!(method::OCBA, problem::Problem{Ty}, iteration::Int, state::OCBAState) where {Ty}
    X= state.X
    sim = problem.objective
    #meana and variance
    transform!(X, AsTable(:).=>ByRow.(x->if x.nbrNextSim>0 
                                            getSamplingMeanVariance(sim,x.x1,x.Y,Int(x.nbrNextSim))
                                        else x.μ, x.σ, x.Y    
                                        end).=> :tmp)
    select!(X, :x1, AsTable(:).=>ByRow.([ x->x.tmp[1], x->x.tmp[2], x->vcat(x.Y,x.tmp[3]),x->x.nbrNextSim+x.nbrTotalSim, x-> 0])
                                            .=> [:μ,:σ,:Y,:nbrTotalSim, :nbrNextSim]);
    println("Finish take samples")
    
    # here we count the addition required runs for each simulation i must mention here the reference for this methods
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
            sum1 += σ[i]/σ[j]*((X.μ[indice_best]-X.μ[j])/(X.μ[indice_best]-X.μ[i]))^2
            sum2+= (1/X.σ[i])* (X.σ[i]^2/X.σ[j]^2*((X.μ[indice_best]-X.μ[j])/(X.μ[indice_best]-X.μ[i]))^4)
        end
    end
    println("Finish calculating sums...")
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
    collect(X.x1[indice_best]), X.μ[indice_best]
end
