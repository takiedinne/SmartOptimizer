using Distributions
using ResumableFunctions
using SimJulia
using Random

const J=3 #number of queus (variables)
const μ=20
const δ=1/30
const σ=0.01
const SEED = 150
const λ=[1,1,1,1,1,1,1,1,1,1]#arrival process poison distrubition parameter
const τ=[1,1,1,10,1,50,1,1,1,1]# Priority

Random.seed!(SEED)
const SwitchOverQueueDistribution= Normal(δ,σ)
const ServiceTimeDistribution=Exponential(μ)
const indifferenceSteadycst= 500 #0.6 sec
const nbrOfSteadytest=5

numberOfCustumer=zeros(J)
currentQueue=1
custumerarrivalTime=[]
custumerWaitingTime=[]
for i in 1:J
    push!(custumerarrivalTime,[])
    push!(custumerWaitingTime,[])
end

SteadyStateAchieved=zeros(J)


@resumable function queue(env::Environment, name::Int, switchProcess::Process, K)
    W_old=0
    currentnbrSteady=0
    while true
        try @yield timeout(env,Inf) catch end
        #initialiser mi par la valeur de ki
        m=K[name]
        global currentQueue= name
        #serve the client
        while numberOfCustumer[name] > 0 && m > 0
            push!(custumerWaitingTime[name], now(env)-popfirst!(custumerarrivalTime[name]))
            @yield timeout(env,rand(ServiceTimeDistribution))
            global numberOfCustumer[name]-=1
            global m-=1
            #println("a customer from queue number $name is served...")
        end
        #here we interrupt switch process 
        @yield interrupt(switchProcess)
    end
    
end

#switch process
@resumable function SwitchQueue(env::Environment)
    while true
        try @yield timeout(env, Inf) catch end
        @yield timeout(env,rand(SwitchOverQueueDistribution))
        global currentQueue=currentQueue+1
        if currentQueue>J  global currentQueue =1 end
        @yield interrupt(processesId[currentQueue])
    end
end

@resumable function CostumerArrival(env::Environment, queueId::Int)
    ArrivalDistribution= Poisson(λ[queueId])
    while true
        @yield timeout(env,rand(ArrivalDistribution))
        push!(custumerarrivalTime[queueId],now(env))
        numberOfCustumer[queueId]+=1
        #=println("CostumerArrival: a new customer is come to the queue number $queueId ...")
        if SteadyStateAchieved[queueId]==1
            break
        end=#
    end
end

@resumable function start_sim(sim::Environment, K)
    
    custumerarrivalTime=[]
    custumerWaitingTime=[]
    for i in 1:J
        push!(custumerarrivalTime,[])
        push!(custumerWaitingTime,[])
    end
    switchProcess= @process SwitchQueue(sim)
    for i in 1:J
        proc=@process CostumerArrival(sim,i)
    end
    @yield timeout(sim,10)
    #queue processes
    for i in 1:J
        proc=@process queue(sim,i,switchProcess,K)
        push!(processesId, proc)
    end
    @yield interrupt(processesId[1])
    # wake up the first queue
end
function sim_GG1K(K)
    
    global processesId=[]
    sim = Simulation()
    @process start_sim(sim,K)
    #println("Simulation is started pour: $K...")
    run(sim,2000)
    #println("Simulation is finished...")
    sumMean= 0
    sumWeight=0
    for i in 1:J
        sumMean+=mean(custumerWaitingTime[i])*τ[i]
        sumWeight+=τ[i]
    end
    sumMean/sumWeight
end
