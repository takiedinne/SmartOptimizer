using Distributions
using ResumableFunctions
using SimJulia
using Random
const J=10 #number of queus (variables)
const μ=20
const δ=1/30
const σ=0.01
const SEED = 150
const λ=[1,1,1,1,1,1,1,1,1,1]#arrival process poison distrubition parameter
const τ=[1,1,1,10,1,50,1,1,1,1]# Priority
const capacity = ones(J).*20 # size max for the queues
Random.seed!(SEED)
const SwitchOverQueueDistribution= Normal(δ,σ)
const ServiceTimeDistribution=Exponential(μ)

numberOfCustumer=zeros(J)
currentQueue=1
custumerarrivalTime=[]
custumerWaitingTime=[]

SteadyStateAchieved=zeros(J)

mutable struct customer
    enterToSystemTime
    exitFromSystemTime
    waitingTime
end

mutable struct Queue
    id :: Int16
    customerInQueue::Array{customer,1}
    servedCustomer::Array{customer,1}
    capacity::Int
    full::Bool
    empty::Bool
    process
end
@resumable function queue_process(env::Environment, server::Resource, queue::Queue, customer_arrival_proc::Process, K)
    while true
        #println("queue process")
        @yield request(server)# waiting our turn
       # println(" the server is now processing the queue N° ", queue.id)
        if length(queue.customerInQueue) > 0 
            m = K #the decision variable
            #serve the client
            
            while length(queue.customerInQueue) > 0 && m > 0
                #println("a customer from queue number $name is goint to serve...")
                customerToServed = popfirst!(queue.customerInQueue)
                customerToServed.exitFromSystemTime = now(env)
                customerToServed.waitingTime = customerToServed.exitFromSystemTime - customerToServed.enterToSystemTime
                push!(queue.servedCustomer, customerToServed)
                #check if the customer arrivel is on sleep
                
                if queue.full 
                    @yield interrupt(customer_arrival_proc)
                    queue.full = false
                end
                @yield timeout(env,rand(ServiceTimeDistribution))
               # println(" a custumer is served from queue n° ",queue.id)
                m -= 1
            end
            # here we simulate the swith between the queues
            #println(" the server is switching")
            @yield timeout(env,rand(SwitchOverQueueDistribution))
            @yield release(server)
        else
            @yield release(server)
            queue.empty = true
            try @yield timeout(env, Inf) catch end 
            queue.empty = false
        end
    end
    
end

@resumable function CostumerArrival(env::Environment, queue::Queue)
    ArrivalDistribution= Poisson(λ[1])
    while true
        @yield timeout(env,rand(ArrivalDistribution))
        if length(queue.customerInQueue) < queue.capacity
            #println(" a customer is arriving to the queue n° ", queue.id)
            c =  customer(now(env), NaN, NaN)
            push!(queue.customerInQueue, c)
            if queue.empty
                @yield interrupt(queue.process)
                queue.empty = false
            end
        else
            queue.full = true
            #println(" ++++++++++++++ the customer arrival is sleeping ", queue.id)
            try @yield timeout(env, Inf) catch end
            #println(" ************** the customer arrival is woken up", queue.id)
        end
    end
end

function sim_GG1K(K)
   # println("simulating $K")
    sim = Simulation()
    # initialize the server
    server = Resource(sim, 1)
    # initialise the queues queues process and arrival customers  
    queues_array = []
    for i in 1:J
        queue = Queue(i, customer[], customer[], capacity[i], false, false, NaN)        
        proc = @process CostumerArrival(sim,queue)
        queueProc = @process queue_process(sim, server, queue, proc, K[i])
        queue.process = queueProc
        push!(queues_array, queue)
    end

    run(sim, 480) # 8 hours

    sumMean = 0
    sumWeight = 0
    for i in 1:J
        if length(queues_array[i].servedCustomer) > 0
            #count the mean of this queue
            sumWaitingTime = 0
            for c in queues_array[i].servedCustomer
                sumWaitingTime += c.waitingTime
            end
            numberOfCustumer = length(queues_array[i].servedCustomer)
            sumMean += sumWaitingTime / numberOfCustumer *τ[i]
            sumWeight += τ[i]
        end
    end
    sumMean/sumWeight
end
