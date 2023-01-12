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
using Plots

Random.seed!(123) # Set the seed for repeatability between test runs

global warehouseNbr = 1
global plantNbr = 1
global run_time = 5000 
global α = 0.9 # customer level (constraints)
global C = ones(10)

global customerOrdersDist = Poisson(1.1)
global orderQuantityRange = 3:9
global serviceLevel = 0.9 # α in the papers 
global productionDurationDist = Exponential(0.1)
global transportationDuration = 3

global warehouses 
global plants 

@resumable function customer_arrival(env::Environment, plant::Resource, id_warehouse::Integer)
    while true
        #simulate laps of time between two orders
        interdemand_time = rand(customerOrdersDist)
        order_quantity = rand(orderQuantityRange)
        backlogged_quantity = 0
        @yield timeout(env, interdemand_time)
        #println("warehouse $id_warehouse: customer arrive at ", now(env), " and order $order_quantity...")
        #check if the we could handle the order immediatley
        if order_quantity <= warehouses.inventory[id_warehouse]
            warehouses.customerDirectlyProcessedNbr[id_warehouse] += 1
            warehouses.inventory[id_warehouse] -= order_quantity
            #println("---> server directely")
        else
            #here we do a backlog
            backlogged_quantity = warehouses.inventory[id_warehouse] > 0 ? order_quantity - warehouses.inventory[id_warehouse] : order_quantity
            warehouses.inventory[id_warehouse] = 0
            warehouses.backlogged_quantity[id_warehouse] += backlogged_quantity
            #println("---> the order is backlogged")
        end
        #check if the current level is less than the reorder level
        inventory_level = warehouses.inventory[id_warehouse] - warehouses.backlogged_quantity[id_warehouse] + warehouses.pending_quantity[id_warehouse]
        if inventory_level < warehouses.reorder_level[id_warehouse]
            order_quantity = warehouses.Q[id_warehouse] * 
                                    (trunc((warehouses.reorder_level[id_warehouse] - inventory_level) 
                                        / warehouses.Q[id_warehouse]) + 1)
            warehouses.pending_quantity[id_warehouse] += order_quantity
            @process replenishement_order(env, plant, Int64(order_quantity), id_warehouse)
        end        
        warehouses.totalCustomerNbr[id_warehouse] += 1
    end
end
@resumable function replenishement_order(env::Environment, plant::Resource, order_quantity::Integer, warehouseId::Integer)
    #println("replenishment order of $order_quantity form warhouse $warehouseId at ", now(env))
    productionDuration = sum(rand(productionDurationDist, order_quantity))
    #production phase
    @yield request(plant)
        #println("---> producting for warhouse $warehouseId ...")
        @yield timeout(env, productionDuration)
        #println("---> production done for warhouse $warehouseId ...")
    @yield release(plant)
    #transportation phase
    #println("---> begin transportation for warhouse $warehouseId")
    @yield timeout(env, transportationDuration)
    #println("---> production have been arrived at warhouse $warehouseId ...")
    #replinishement arrival phase
    warehouses.pending_quantity[warehouseId] -= order_quantity
    if order_quantity < warehouses.backlogged_quantity[warehouseId]
        warehouses.backlogged_quantity[warehouseId] -= order_quantity
    else
        overflow_quantity =  order_quantity - warehouses.backlogged_quantity[warehouseId]
        warehouses.backlogged_quantity[warehouseId] = 0
        warehouses.inventory[warehouseId] += overflow_quantity
    end
end

@resumable function inventory_observer(env::Environment, id_warehouse::Integer)
    while true 
        @yield timeout(env, 1)
        level = warehouses.inventory[id_warehouse]
        push!(warehouses.on_hand_level_observation[id_warehouse], level)
    end
end

function inventoryRoutingProblem(X)
    global warehouses = DataFrame(idWarehouse = collect(1:warehouseNbr),
                             inventory = X[warehouseNbr+1:end],
                             reorder_level =X[1:warehouseNbr],
                             Q = X[warehouseNbr+1:end],
                             backlogged_quantity = zeros(warehouseNbr),
                             pending_quantity  = zeros(warehouseNbr),
                             on_hand_level_observation = [[] for _ in 1:warehouseNbr],
                             customerDirectlyProcessedNbr=zeros(warehouseNbr),
                             totalCustomerNbr =zeros(warehouseNbr)
                             )
    sim = Simulation()

    #plants 
    plants = [Resource(sim, 1) for _ in 1:plantNbr]
    for i in 1:warehouseNbr
        @process customer_arrival(sim, plants[1], i)
        @process inventory_observer(sim, i)
    end

    run(sim, run_time)
    #check that this solution complies with the constraint p(R)>= α
    customerLevel = sum(warehouses.customerDirectlyProcessedNbr) / sum(warehouses.totalCustomerNbr)
    
    #plote the inventory level
   #= plot()
    for i in 1:warehouseNbr
        x = collect(1:100:length(warehouses.on_hand_level_observation[i]))
        y = [warehouses.on_hand_level_observation[i][j] for j in x] 
        display(plot!(x, y, seriestype = [:bar]))
    end=#
    if customerLevel >= α 
        warehouses.mean = mean.(warehouses.on_hand_level_observation)
        fit = sum( warehouses.mean[i] * C[i] for i in 1:warehouseNbr)
        return fit
    else 
        return 10^5 # penality
    end
end
