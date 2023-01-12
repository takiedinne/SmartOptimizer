#=
this is ranking and selecting procedure from handbook of simulation optimization

=#
using Distributions
using Combinatorics
using DataFrames
using CSV
using Plots ;
using Conda 
pyplot()
import Pkg
#global variable
"PyPlot" ∈ keys(Pkg.installed())
normalDist=Normal(2.3,16)
Objective_function(X)= X[1]^3-12*X[2]^2+16 + rand(normalDist)
f(x,y)=x^3-12*y^2+16 + rand(normalDist)
D1=collect(1:1:3)
D2=collect(1:1:3)
X=collect(Iterators.product(D1,D2))
#reshap to one column
X=reshape(X,(:,1))
Y=[]
for i in eachindex(X)
    push!(Y,Objective_function(X[i]))
end
Y
plot(D1,D2,f,st=:surface)

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

                                    
function getSamplingresultsAndMean(obj::Function,X::Tuple, n0::Int)
    Z=[]
    for i in 1:n0
        push!(Z,obj(X))
    end
    μ=mean(Z)
    μ, Z
end

function S²_il(X::DataFrame)
    S=zeros(nrow(X),nrow(X))
    for i in 1:nrow(X)
        x1=X[i,:Y]
        for l in i+1:nrow(X)
            x2=X[l,:Y]
            sum=0
            diffMean=(mean(x1)-mean(x2))
            for j in eachindex(x1)
               sum+= (x1[j]-x2[j]-diffMean)^2
            end
            S[i,l]=sum/(length(x1)-1)
            S[l,i]=sum/(length(x1)-1)
        end
    end
    S
end
#rank selecting algorithm
function RankSelect(obj::Function,D...)
    #parameters
    n0=80
    δ=1 #indifference zone
    α=0.95 # intervale of confidence
    #combine all the solution
    X=collect(Iterators.product(D...))
    #reshap to one column
    X=reshape(X,(:,1))
    k= length(X)# size of the problem
    η=0.5*((2*α/(k-1))^(-2/(n0-1))-1)
    h²=2*η*(n0-1)
    #using dataframe to store the solution, mean, variance
    X=convert(DataFrame,X)
    #meana and variance
    transform!(X, AsTable(:).=>ByRow.(x->getSamplingresultsAndMean(obj,x.x1,n0)).=> :tmp)
    select!(X, :x1,:tmp.=>ByRow.([ x->x[1], x->x[2]]).=> [:μ,:Y])
    r=n0
    while r<1000
        S=S²_il(X)
        I_old=deepcopy(X)
        X= DataFrame()
        for i in 1: k
            bool=true
            for j in 1:k 
                
                w=δ/(2*r)*(trunc(S[i,j]*h²/δ^2-1)+1)
                if I_old[i,:μ]>I_old[j,:μ]+w 
                    bool=false
                end
            end
            if bool
                push!(X,I_old[i,:])
            end       
        end
        if nrow(X)==1
            return X[1,:]
        else
            #taking new simulation
            for i in 1:nrow(X)
                X[i,:μ]=(obj(X[i,:x1])+r*X[i,:μ])/(r+1)
            end
            r+=1
        end
    end
end
X=RankSelect(Objective_function,D1,D2)

