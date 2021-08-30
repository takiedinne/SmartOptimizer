"""
    MTO: mark to order problem is the schudling problem in the manifactoring area 
    this is is the very simple example extracted from [1]
    [1]: Song, D. P., Hicks, C., & Earl, C. F. (2006).
         An ordinal optimization based evolution strategy to schedule complex 
         make-to-order products. International Journal of Production Research, 44(22),
         4877–4895. https://doi.org/10.1080/00207540600620922 
"""

using Distributions
using ResumableFunctions
using SimJulia
using Random
using DataFrames

#i will represent the tree as matrix because in the first case we have only 8 nodes 
# if i can gather another scenarion a would be use anothe package to manage the tree structure namely Abstract tree, 
# or DataStractures...etc.
const product_structure = [0 1 1 1 0 0 0 0;
                           0 0 0 0 1 1 0 0;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 1 1;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0 0;] # precedence matrix

# assignment of each operation to a specific machine 
const machine_operation_list = [1, 1, 2, 1, 2, 2, 3, 3] #machine_operation_list[i]=> the index of the machine for the operation i
const μ = [20, 20, 10, 20, 10, 10, 10, 10] #the means of processing time for each operation
const nbr_of_op = 8
const nbr_of_machines = 3
const due_date = 100

global operations_df
global machines_df

global machines_proc
global scheduler_proc

global finish = false
 
#= machine process
    this process imitate the behaviour of a machine. a machine has a queue at each iteration take the
    operation with respect with priority policy (this version work with FIFO policy), then made thisoperation 
    if there is no operation in the queue so the machine go to sleep state  
=#
@resumable function machine(env::Environment, id_machine::Integer)
    #println("Starting machine $id_machine")
    # this process represent tha machine 
    # at first  we assume that the machines work with FIFO policy
    while !finish || !isempty(machines_df.queue[id_machine])
        # turn off the machine if there is no operation in the queue
        if isempty(machines_df.queue[id_machine])
            global machines_df.state[id_machine] = false # sleep
            try @yield timeout(env, Inf) catch end
            machines_df.state[id_machine] = true # awake
        end
        #get the next operation to be processed by te machine and delete it from the queue
        #next_operation = priorityRule(machines_queues[id_machine])
        next_operation = popfirst!(machines_df.queue[id_machine])
        processing_distrubution = Exponential(μ[next_operation])
        working_duration = rand(processing_distrubution)
        operations_df.duration[next_operation] = working_duration
        # processing the opertaion
        println("machine $id_machine : processing the operation $next_operation at ", now(env), " for $working_duration")
        
        @yield timeout(env, working_duration) 
        current_time = now(env)
        println("machine $id_machine : finish processing the operation $next_operation at ", current_time)
        operations_df.completion_time[next_operation] = current_time
        #update the available time pour the parrent
        update_available_time(next_operation, current_time) 
        try @yield interrupt(scheduler_proc) catch end # notify the scheduler_proc to chack availabe time
    end
end
#= scheduler process
    this process is responsible to add the operation to the correspond queue 
    that after checking that their start time and available time are arrived
=#
@resumable function scheduler_process(env::Environment)
    # this process is responsible to add the operation to the queue of their corespond machine
    # with respect to the start time and the available time
    remaining_operation = collect(1:nbr_of_op)

    while !isempty(remaining_operation)
        # at each iteration the process check if the remaining operation are ready to be processed
        for i in remaining_operation
            current_time = now(env)
            if operations_df.available_time[i] <= current_time && operations_df.start_time[i] <= current_time
                correspond_machine = operations_df.machine[i]
                # here we add this op to its corresponds machine's queue
                println("add operation $i to the machine $correspond_machine's queue at $current_time") 
                push!(machines_df.queue[correspond_machine], i)
                # delete this operation from the remaing machine_operation_list
                filter!(x-> x!= i, remaining_operation)
                # notify the correspondant machine if it is sleeping
                if ! machines_df.state[correspond_machine] #check if the machine is sleep or not
                    try @yield interrupt(machines_df.machine_proc[correspond_machine]) catch end 
                end
            end
        end
        #println("scheduler_process sleep")
        try @yield timeout(env , Inf) catch end 
        #println("scheduler_process wake up")
    end 
    finish = true
end
#= start time survey process
    this process is responsible notify the scheduler that a nex operation is ready to dispatch 
    (their start time is arrive ) 
=#
@resumable function start_time_survey(env::Environment)
    #println("Starting start_time_survey process")
    remaining_operation = copy(operations_df.start_time)
    while !isempty(remaining_operation)
        #println("start_time_survey iteration begin")
        current_time = now(env)
        sleep_time = minimum(remaining_operation) - current_time
        if sleep_time > 0
            try @yield timeout(env, sleep_time) catch end 
        end
        try @yield interrupt(scheduler_proc) catch end 
        deleteat!(remaining_operation, argmin(remaining_operation))
        #println("start_time_survey iteration end")
    end
end

#=
    this function resposible of updatinh the available_time after handling one operation
=#
function update_available_time(op_id::Integer, current_time)
    # this function will be invoked after that the op_id finished to update the available time for the parent
    # get the parrent of this operation from the product structure matrix 
    # as we have tree structure so there is only one parrent
    parent_ind = argmax(product_structure[:, op_id]) # necessary only one will be equal to one 
    
    # check if all the offsprings are completed 
    available_flag = true
    for i in 1:size(product_structure)[1]
        #check all childs for this parrent operation if they finish
        if product_structure[parent_ind, i] == 1 && operations_df.completion_time[i] === Inf
            available_flag = false
        end
    end

    if available_flag 
        operations_df.available_time[parent_ind] = current_time
    end
end

function MTO_product(start_time)
    @assert length(start_time) == nbr_of_op "decision variable must be on the size of the problem..."
    sim = Simulation() 

    global operations_df= DataFrame(id_op=collect(1:nbr_of_op),#id operation 
                                    μ = μ, # mean processing time for the operation 
                                    machine = machine_operation_list, # correspanding machine for each operation
                                    start_time = start_time, 
                                    available_time = [ sum(product_structure[i,:]) == 0 ? 0 : Inf for i in 1:nbr_of_op], #time that this operation is available to processing equal max(si, cj where j is all the childs of operation i)
                                    completion_time = ones(nbr_of_op).*Inf, # completion time
                                    duration = zeros(nbr_of_op),
                                    holding_cost = zeros(nbr_of_op) # will be calculated after the simulation run
                                    )
   
    global machines_df = DataFrame(id_machine = collect(1:nbr_of_machines),
                                   queue= [ Integer[] for i in 1:nbr_of_machines] ,
                                   state = trues(nbr_of_machines),
                                   machine_proc = [ @process machine(sim,i) for i in 1:nbr_of_machines ] )
      
    global scheduler_proc = @process scheduler_process(sim)
    @process start_time_survey(sim)
    run(sim)
    #computing the fiteness function for the problem.
    # from the paper f(x) = ∑ h_i(a_fatherde i - c_i)  for i ∉ root (the product structure can have sevarale roots)
    #                       + ∑ h_i max(d_i - c_i, 0)  for i ∈ root (root node)
    #                       + ∑ p_i max(c_i - d_i, 0)  for i ∈ root
    # h_i is the unit time holding cost for this paper as i undestand is the sum of duration
    #    for the operation i and its predecessors multiplied bu 0.01
    # p_i unit time tardiness cost equal double of h_i
    # the due date for the product is 100 time unit.

    # first calculate h_i for all operation
    calculate_operations_holding_cost!(1) # as the papameters are global so we don't need to pass them to the function 
    #the first sum in the fitness function
    WIP_holding_cost = sum( i.holding_cost * 
                            (operations_df.available_time[argmax(product_structure[:, i.id_op])] - i.completion_time)
                            for i in  eachrow( filter(row -> maximum(product_structure[:, row.id_op]) != 0 , operations_df) ) )
    #here we assume the product has only one final node and always is the first node 
    earliness_cost = operations_df.holding_cost[1] * max(due_date - operations_df.completion_time[1], 0)
    tardiness_cost = 2 * operations_df.holding_cost[1] * max(operations_df.completion_time[1] - due_date , 0)
    return WIP_holding_cost + earliness_cost + tardiness_cost
end

function calculate_operations_holding_cost!(node_id::Integer)
    
    h = operations_df.duration[node_id]
    for i in findall(j-> j == 1, product_structure[node_id, :] )
       h += calculate_operations_holding_cost!(i)
    end
    operations_df.holding_cost[node_id] = h
end