#=
    production distrubution system is modeled as described in [1] and  [2].
    this system contains manifactoring plants and several warehouses.
    each warehouse receives orders which demand X product (aleatoire variable). 
    if the inventory of the on-hand warehouse is enough to process the order so it does that
    other wise the order of the custumer can be split and the remaining order is backlogged. 
    in the last case, a replenishement order in proceed to the plant.
    each warehouse contains two decision variables: 
    (R => base stock limit and Q => all the replinshement request must be with size n*Q ) 
=#
using ResumableFunctions
using Distributions
using Random
using SimJulia
using DataFrames

Random.seed!(123) # Set the seed for repeatability between test runs

global warehouseNbr = 1
global plantNbr = 1
global run_time = 5000
global α = 0.9 # customer level
global C = ones(10)

global customerOrdersDist = Poisson(1.6)
global orderQuantityRange = 3:9
global serviceLevel = 0.9 # α in the papers 
global productionDurationDist = Exponential(0.1)
global transportationDuration = 3

global warehouses 
global plants 

@resumable function customersArrivalProcess(env::Environment, warhouseId::Integer)
    while true
        #simulate laps of time between two orders
        waitingtime = rand(customerOrdersDist)
        orderQuantity = rand(orderQuantityRange)
        @yield timeout(env, waitingtime)
        # add the order
        push!(warehouses.pendingOrder[warhouseId], orderQuantity)
        #println("customer arriving to the warhouse $warhouseId at ", now(env))
        if !warehouses.state[warhouseId]
            @yield interrupt(warehouses.processId[warhouseId])
        end
    end
end

# use a boolean variable to precice if the customer is prossesed directly or he was waiting
@resumable function warehouseProcess(env::Environment, warehouseId::Integer, R::Integer)
    while true
        if (isempty(warehouses.backloggedQuantities[warehouseId]) && isempty(warehouses.pendingOrder[warehouseId])) ||
                                                        warehouses.inventory[warehouseId] == 0
            # suspend this process
            #println("the warehouse $warehouseId is turn off at ", now(env))
            warehouses.state[warehouseId] = false
            try @yield timeout(env, Inf) catch end
            warehouses.state[warehouseId] = true # state is on
            #println("the warehouse $warehouseId is turn on at", now(env))
        end
         # check the warehouse position to eventual order from the plant
         PI = warehouses.inventory[warehouseId] - sum(warehouses.pendingOrder[warehouseId]) + 
         sum(warehouses.replenishmentOrder[warehouseId])
     
        if PI < R
            #generate a new orser to the manifactoring plant
            # here we assume that Q in the paper is 1 because after the testes proceeeded by the authers they found 
            # that Q= 1 is always the optimal so we generate the order  equal the difference between the PI and R
            newOrder = R - PI + 1
            push!(plants.queueQuantities[1], newOrder)
            push!(warehouses.replenishmentOrder[warehouseId], newOrder)
            push!(plants.queueWarhouses[1], warehouseId)
            if !plants.state[1]
                @yield interrupt(plants.processId[1])
            end
            #println("warehouse $warehouseId : repelenishment order at ", now(env))
        end
        #check if there is backlogged
        while !isempty(warehouses.backloggedQuantities[warehouseId]) && warehouses.inventory[warehouseId] > 0
            # we are in the case where the process was suspended because of unsuffisant inventory
            currentOrder = warehouses.backloggedQuantities[warehouseId][1] # the first order
            if currentOrder <= warehouses.inventory[warehouseId]
                warehouses.inventory[warehouseId] -= currentOrder
                warehouses.totalInventory[warehouseId] += currentOrder
                warehouses.totalCustomerNbr[warehouseId] += 1
                popfirst!(warehouses.backloggedQuantities[warehouseId])
                #println("warehouse $warehouseId : backlogged customer is served at ", now(env))
            else 
                #backlogged again
                currentOrder -= warehouses.inventory[warehouseId]
                warehouses.totalInventory[warehouseId] += warehouses.inventory[warehouseId]
                warehouses.inventory[warehouseId] = 0
                warehouses.backloggedQuantities[warehouseId][1] = currentOrder
                #println("warehouse $warehouseId : backlogged again at ", now(env))
            end
        end
        
        while !isempty(warehouses.pendingOrder[warehouseId]) 
                currentOrder = popfirst!(warehouses.pendingOrder[warehouseId])
                if currentOrder <= warehouses.inventory[warehouseId]
                    warehouses.inventory[warehouseId] -= currentOrder
                    warehouses.totalInventory[warehouseId] += currentOrder
                    warehouses.totalCustomerNbr[warehouseId] += 1
                    warehouses.customerDirectlyProcessedNbr[warehouseId] += 1
                    #println("warehouse $warehouseId : customer is served imediately at ", now(env))
                else
                    #backlogged
                    currentOrder -= warehouses.inventory[warehouseId]
                    warehouses.totalInventory[warehouseId] += warehouses.inventory[warehouseId]
                    warehouses.inventory[warehouseId] = 0
                    push!(warehouses.backloggedQuantities[warehouseId], currentOrder)
                    #println("warehouse $warehouseId : a customer is backlogged at ", now(env))
                end
        end
       
    end
end

@resumable function plantProcess(env::Environment, plantId::Integer)
    while true
        if isempty(plants.queueQuantities[plantId])
            #println("plant $plantId is turned off ", now(env))
            plants.state[plantId] = false
            try @yield timeout(env, Inf) catch end 
            plants.state[plantId] = true
            #println("plant $plantId is turned on at ", now(env))
        end
        currentOrder = popfirst!(plants.queueQuantities[plantId])
        warehouseOrdered = popfirst!(plants.queueWarhouses[plantId])

        productionDuration = sum(rand(productionDurationDist, currentOrder))
        @yield timeout(env, productionDuration)
        #println("plant $plantId : production of $currentOrder products is finished at ", now(env))
        # create a transportation process to arrive to the warehouse
        @process transportationProcess(env, currentOrder, warehouseOrdered)
    end
end
@resumable function transportationProcess(env::Environment, quantity::Integer, warehouseId::Integer)
    #printstyled("replishment trosportation\n", color=:blue)
    @yield timeout(env, transportationDuration)
    warehouses.inventory[warehouseId] += quantity
    popfirst!(warehouses.replenishmentOrder[warehouseId])
    if !warehouses.state[warehouseId] 
        @yield interrupt(warehouses.processId[warehouseId])
    end
    #println("replenishment of $quantity products is arrived to warehouse $warehouseId at ", now(env))
end

function inventoryRoutingProblem(R)
    global warehouses = DataFrame(idWarehouse = collect(1:warehouseNbr),
                             inventory= R , backloggedQuantities= [[] for i in 1:warehouseNbr],
                             pendingOrder= [Integer[] for i in 1:warehouseNbr], 
                             replenishmentOrder=[Integer[] for i in 1:warehouseNbr], 
                             totalInventory = zeros(warehouseNbr),
                             state=trues(warehouseNbr), customerDirectlyProcessedNbr=zeros(warehouseNbr),
                             totalCustomerNbr =zeros(warehouseNbr), 
                             processId= Array{Process, 1}(undef, warehouseNbr), 
                             meanInventory =zeros(warehouseNbr))
    global plants = DataFrame(idPlant = collect(1:plantNbr), 
                                queueQuantities = [Integer[] for i in 1:plantNbr],
                                queueWarhouses = [Integer[] for i in 1:plantNbr], 
                                state= trues(plantNbr),processId= Array{Process, 1}(undef, plantNbr) )
    sim = Simulation()
    for i in 1:warehouseNbr
        @process customersArrivalProcess(sim, i)
    end
    for i in 1:warehouseNbr
        proc = @process warehouseProcess(sim, i, R[i])
        warehouses.processId[i] = proc
    end
    for i in 1:plantNbr
        proc = @process plantProcess(sim, i)
        plants.processId[i] = proc
    end 

    run(sim, run_time)
    #check that this solution complies with the constraint p(R)>= α
    customerLevel = sum(warehouses.customerDirectlyProcessedNbr)/ sum(warehouses.totalCustomerNbr)
    
    if customerLevel >= α 
        warehouses.mean = [ (warehouses.totalInventory[i] + warehouses.inventory[i])/ run_time for i in 1: warehouseNbr]
        fit = sum( warehouses.mean[i] * C[i] for i in 1:warehouseNbr)
        return fit
    else 
        return Inf
    end
end
