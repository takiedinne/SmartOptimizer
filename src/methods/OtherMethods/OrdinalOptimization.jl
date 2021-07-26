#=
this method is implemented from ORDINAL OPTIMIZATION APPROACH TO STOCHASTIC SIMULATION
OPTIMIZATION PROBLEMS AND APPLICATIONS
=#
using Flux
using SimJulia
using IterTools: ncycle
using DataFrames, Evolutionary
include("../../../simulations/GG1K_simulation.jl")

const M=500 # nbr of solution to build the ANN model
const I=1000 #nbre of population in the genetic algorithme
const N=1000 #nbr od solution considred in the third step
const s=35 #nbr for the last step
const Lm=1
const Ls=1
const probabilitie_cross_over=0.7
const probability_of_mutation=0.02

function Cross_over(x,y)
    c=rand(1:length(x))
    child=vcat(x[1:c],y[c+1:length(x)])
    return child
end

function mutation(x,UpBound)
    place_to_mutated=rand(1:length(x))
    x[place_to_mutated]=rand(1:UpBound)
end

# the Genetic algorithme implementation
function genetic_algorithme(fitness_fun,population, nbr_generation::Int,UpBound::Int)
    println("begin the genetic algo ...")
    current_nbr_generation=0
    #calculate fitness for all the population
    population_dataFrame=DataFrame(x=[], fit=Float64[])
    for i in 1:size(population)[1]
        push!(population_dataFrame, (population[i,:], fitness_fun(population[i,:])[1]))
    end
    while current_nbr_generation<nbr_generation
        i1 , i2=roulette(population_dataFrame.fit,2)
        parent1=population_dataFrame.x[i1]
        parent2=population_dataFrame.x[i2]
        child=copy(parent1)# if there is no cross over so we work with the parrent 1
        if rand()<probabilitie_cross_over
            child=Cross_over(parent1,parent2)
        end
        if rand()<probability_of_mutation
            mutation(child,UpBound)
            current_nbr_generation+=1
            println("we already generate $current_nbr_generation")
        end
        push!(population_dataFrame,(child, fitness_fun(child)[1]))
    end
    # here we sort the population according to their fitness values
    sort!(population_dataFrame,[:fit])
    population_dataFrame
end


function OrdinalOptimization(sim::Function, NbrVar::Int, UpBound::Int)
    #step 1 create a suurogate model of oure simulation using ANN
    #we get data set we generate 500 condidate solution uniformaly
    X_ANN=rand(collect(1:1:UpBound),M)
    for i in 2:NbrVar
        X_ANN= hcat(X_ANN, rand(collect(1:1:UpBound),M) )
    end
    #evaluate all this sampling
    Y=[]
    for i in 1:M
        push!(Y,sim(X_ANN[i,:]))
    end
    #here we train our ANN
    m = Chain(
            Dense(NbrVar, 40,relu),
            Dense(40, 1,relu)
        )

    loss(x) = Flux.crossentropy(m(x.images), x.labels)
    accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))
   
    train_loader = Flux.Data.DataLoader((images=X_ANN', labels=Y), batchsize=2, shuffle=true)
    evalcb = () ->println("trainning...")
    ps = Flux.params(m)
    Flux.train!(loss, ps,  ncycle(train_loader,5), ADAM(), cb=Flux.throttle(evalcb, 2))
    # second step
    #first generate I random population
    X=rand(collect(1:1:UpBound),I)
    for i in 2:NbrVar
        X= hcat(X, rand(collect(1:1:UpBound),I) )
    end
    X=genetic_algorithme(m,X,20,20)
    # third step
    #take N solution ffrom the pool of genetic algorithme
    X=X[1:N,:]
    #simulate each solution Lm timeout
    for row in eachrow(X)
        sum=0
        for i in 1:Lm
            sum+=sim_GG1K(row.x)
        end
        row.fit=sum/Lm
    end
    sort!(X,[:fit])
    #fourth step
    X=X[1:s,:]
    for row in eachrow(X)
        sum=0
        for i in 1:Ls
            sum+=sim_GG1K(row.x)
        end
        row.fit=sum/Ls
    end
    sort!(X,[:fit])
    X[1,:]
end
startTime=time()
m=OrdinalOptimization(sim_GG1K,10,20)
endTime=time()
delay=(endTime-startTime)/60
m

#=
m = Chain(
            Dense(10, 20,relu),
            Dense(20, 1,relu)
        )

    loss(x) = Flux.crossentropy(m(x.images), x.labels)
    accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))
   
    train_loader = Flux.Data.DataLoader((images=X_ANN', labels=Y), batchsize=2, shuffle=true)
    loss(x,y) = Flux.crossentropy(m(x), y)
    data=zip((X_ANN...),Y)
    first(data)
    loss(first(data)[1],first(data)[2])
    evalcb = () ->println("trainning...")
    ps = Flux.params(m)
    Flux.train!(loss, ps,  train_loader, ADAM(), cb=Flux.throttle(evalcb, 2))
    # second step=#
