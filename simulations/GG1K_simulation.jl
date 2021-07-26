using Distributions
using ResumableFunctions
using SimJulia
using Random

const J=10 #number of queus (variables)
const μ=20
const δ=1/30
const σ=0.01
const SEED = 150
const λ=[7,7,7,7,7,7,7,7,7,7]#arrival process poison distrubition parameter
const τ=[1,1,1,10,1,50,1,1,1,1]# Priority

Random.seed!(SEED)
const SwitchOverQueueDistribution= Normal(δ,σ)
const ServiceTimeDistribution=Exponential(μ)

numberOfCustumer=zeros(J)
currentQueue=1
custumerarrivalTime=[]
custumerWaitingTime=[]

SteadyStateAchieved=zeros(J)


@resumable function queue(env::Environment, name::Int, switchProcess::Process, K)
    while true
        #println("queue process")
        try @yield timeout(env,Inf) catch end
        #initialiser mi par la valeur de ki
        m=K[name]
        #global currentQueue= name
        #serve the client
        while length(custumerarrivalTime[name]) > 0 && m > 0
            #println("a customer from queue number $name is goint to serve...")
            push!(custumerWaitingTime[name], now(env) - popfirst!(custumerarrivalTime[name]))
            
            @yield timeout(env,rand(ServiceTimeDistribution))
            
            global numberOfCustumer[name] -= 1
            global m-=1
            #println("a customer from queue number $name is served...")
        end
        #here we interrupt switch process
        #println(" tring to interrupt switch queue")
        @yield interrupt(switchProcess)
        #println(" end interrupting switch queue")
    end
    
end

#switch process
@resumable function SwitchQueue(env::Environment)
    while true
        #println("SwitchQueue process")
        try @yield timeout(env, Inf) catch end
        @yield timeout(env,rand(SwitchOverQueueDistribution))
        global currentQueue = currentQueue + 1
        if currentQueue > J  global currentQueue = 1 end

        @yield interrupt(processesId[currentQueue])
        
    end
end

@resumable function CostumerArrival(env::Environment, queueId::Int)
    ArrivalDistribution= Poisson(λ[queueId])
    while true
        #println("CostumerArrival process")
        @yield timeout(env,rand(ArrivalDistribution))
        push!(custumerarrivalTime[queueId],now(env))
        numberOfCustumer[queueId] += 1
    end
end

@resumable function start_sim(sim::Environment, K)
    switchProcess= @process SwitchQueue(sim)
    for i in 1:J
        @process CostumerArrival(sim,i)
    end 
    #queue processes
    for i in 1:J
        proc=@process queue(sim,i,switchProcess,K)
        push!(processesId, proc)
    end
    @yield interrupt(processesId[ 1])
end
function sim_GG1K(K)
    global custumerarrivalTime
    global custumerWaitingTime
    custumerWaitingTime = []
    custumerarrivalTime = []
    for i in 1:J
        push!(custumerarrivalTime,[])
        push!(custumerWaitingTime,[])
    end
    global processesId=[]
    sim = Simulation()
    @process start_sim(sim,K)
    #println("Simulation is started pour: $K...")
    run(sim, 2500)
    #println("Simulation is finished...")
    sumMean= 0
    sumWeight=0
    for i in 1:J
        if length(custumerWaitingTime[i]) > 0
            sumMean+=mean(custumerWaitingTime[i])*τ[i]
            sumWeight+=τ[i]
        end
    end
    sumMean/sumWeight
end