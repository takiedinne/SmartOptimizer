#=
this is ranking and selecting procedure from Kim and nelson 2007
there are several problems is to get the rinnot values and also 
the problem to save the result of simulation of the systems to use 
it if there is an addition simulations
=#
using Distributions
using Combinatorics
using DataFrames
using CSV
#global variable
normalDist=Normal(2.3,15)
Objective_function(X)= X[1]^3-12*X[2]^2+16 + rand(normalDist)
D1=collect(1:1:3)
D2=collect(1:1:3)


function getRinnotCst(n0::Int, δ::Int)
    #here we can do switch and case to choose the best file according to the P value
    #here i start by one file corresponding to p=0.90
    table = CSV.File("StudentizedRangeDistrubution.csv") |> Tables.matrix
    if(n0+1-table[1] < 24)
        table[Int(n0+1-table[1]),Int(δ+1)]
    else 
        5.24
    end
end

                                    
function getSamplingMeanVariance(obj::Function,X::Tuple, n0::Int)
    Z=[]
    for i in 1:n0
        push!(Z,obj(X))
    end
    μ=mean(Z)
    σ=var(Z)
    μ, σ
end

#rank selecting algorithm
function RankSelect(obj::Function,D...)
    #parameters
    n0=80
    δ=1
    #combine all the solution
    X=collect(Iterators.product(D...))
    #reshap to one column
    X=reshape(X,(:,1))
    #using dataframe to store the solution, mean, variance
    X=convert(DataFrame,X)
    #meana and variance
    transform!(X, AsTable(:).=>ByRow.(x->getSamplingMeanVariance(obj,x.x1,n0)).=> :tmp)
    select!(X, :x1,:tmp.=>ByRow.([ x->x[1], x->x[2]]).=> [:μ,:σ])
    #get xi from StudentizedRangeDistrubution
    @show ψ= getRinnotCst(n0,δ)
    # for each systeme get the additiona sampling simulation required
    
    for x in eachrow(X)
        N= max(n0,trunc(Int,ψ^2*x.σ/δ^2)+1)
        if N> n0 
            x[:μ],x[:σ] = getSamplingMeanVariance(obj,x.x1,N)
        end
    end
    #return the Tuple with the minumum sampling mean
    X[argmin(X.μ),:x1]
end
X=RankSelect(Objective_function,D1,D2)