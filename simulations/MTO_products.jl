#=
    MTO: mark to order problem is the schudling problem in the manifactoring area 
    this is is the very simple example extracted from [1]
    [1]: Song, D. P., Hicks, C., & Earl, C. F. (2006).
         An ordinal optimization based evolution strategy to schedule complex 
         make-to-order products. International Journal of Production Research, 44(22),
         4877–4895. https://doi.org/10.1080/00207540600620922 
=#

using Distributions
using ResumableFunctions
using SimJulia
using Random
using DataFrames
using LightGraphs
#= I will represent the tree as matrix because in the first case we have only 8 nodes 
 if i can gather another scenarion a would be use anothe package to manage the tree
 structure namely Abstract tree, 
 or DataStractures...etc.=#

# scenario definition
const product_structure = [0 1 1 1 0 0 0 0;
                           0 0 0 0 1 1 0 0;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 1 1;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0 0] # precedence matrix
global product_graph = SimpleDiGraph(product_structure)# assignment of each operation to a specific machine 
const machine_operation_list = [1, 1, 2, 1, 2, 2, 3, 3] #machine_operation_list[i]=> the index of the machine for the operation i
const μ = [20, 20, 10, 20, 10, 10, 10, 10] #the means of processing time for each operation
const nbr_of_op = 8 
const nbr_of_machines = 3
const due_date = 100

global operations_df
global machines_df # machines are represeneted by DataFrame (each machine represeneted by one row)

global machines_proc
global scheduler_proc

global finish = false

global due_date_for_all_operations = zeros(nbr_of_op)
function calculate_due_date(root, g)
    # we assume we have only one 
    nodes_queue = [root] 
    while !isempty(nodes_queue)
        current_node = popfirst!(nodes_queue)
        parent = inneighbors(g, current_node)
        if isempty(parent)
            #here we are in the root node
            due_date_for_all_operations[current_node] = due_date
        else
            parent = parent[1]
            due_date_for_all_operations[current_node] = due_date_for_all_operations[parent] - μ[parent]
        end
        for i in outneighbors(g, current_node)
            push!(nodes_queue, i)
        end
    end
end
calculate_due_date(1, product_graph) # calculate the due time for each operation

global operation_time_remaining = zeros(nbr_of_op)
function calculate_operation_time_remaining(root, g)
    # we assume we have only one 
    nodes_queue = [root] 
    while !isempty(nodes_queue)
        current_node = popfirst!(nodes_queue)
        parent = inneighbors(g, current_node)
        if isempty(parent)
            #here we are in the root node
            operation_time_remaining[current_node] = μ[current_node]
        else
            parent = parent[1]
            operation_time_remaining[current_node] = operation_time_remaining[parent] + μ[current_node]
        end
        for i in outneighbors(g, current_node)
            push!(nodes_queue, i)
        end
    end
end
calculate_operation_time_remaining(1, product_graph) # calculate the due time for each operation

# the priority rules from the same paper 
function FCFS(queue, t)
    # first come first served (according to available time)
    queue[argmin(operations_df.available_time[queue])]
end

function SPT(queue, t)
    # shortest mean processing time
    queue[argmin(μ[queue])]
end

function EDD(queue, t)
    # shortest mean processing time
    queue[argmin(due_date_for_all_operations[queue])]
end

function LWR(queue, t)
    # shortest mean processing time
    queue[argmin(operation_time_remaining[queue])]
end

function LST(queue, t)
    # shortest mean processing time
    queue[argmin(due_date_for_all_operations[queue] .- μ[queue] .- t)]
end

function LCR(queue, t)
    # shortest mean processing time
    queue[argmin((due_date_for_all_operations[queue] .- t) ./ μ[queue])]
end

function EPS(queue, t)
    # shortest mean processing time
    queue[argmin(operations_df.start_time[queue])]
end
global priority_rule_list = [FCFS, SPT, EDD, LWR, LST, LCR, EPS]
#= machine process
    this process imitate the behaviour of a machine. a machine has a queue at each iteration take the
    operation with respect with priority policy (this version work with FIFO policy), then perform
    this operation 
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
            #println("machine $id_machine : is asleep at ", now(env))
        
            try @yield timeout(env, Inf) catch end
            #println("machine $id_machine : is wake up at ", now(env))
            machines_df.state[id_machine] = true # awake
        end
        #get the next operation to be processed by te machine and delete it from the queue
        operation_to_be_processed = machines_df.priority_rule[id_machine](machines_df.queue[id_machine], now(env))
        filter!(x -> x!= operation_to_be_processed, machines_df.queue[id_machine])
        #operation_to_be_processed = popfirst!(machines_df.queue[id_machine]) 
        processing_distrubution = Exponential(μ[operation_to_be_processed])
        working_duration = rand(processing_distrubution)
        operations_df.duration[operation_to_be_processed] = working_duration
        # processing the opertaion
        #println("machine $id_machine : processing the operation $operation_to_be_processed at ", now(env), " for $working_duration")
        
        @yield timeout(env, working_duration) 
        current_time = now(env)
        #println("machine $id_machine : finish processing the operation $operation_to_be_processed at ", current_time)
        operations_df.completion_time[operation_to_be_processed] = current_time
        #update the available time pour the parrent
        update_available_time(operation_to_be_processed, current_time) 
        try @yield interrupt(scheduler_proc) catch end # notify the scheduler_proc to check availabe time
    end
end
#= scheduler process
    this process is responsible to add the operation to the correspond queue 
    that after checking that their start time and available time are arrived
=#

@resumable function scheduler_process(env::Environment)
    
    remaining_operation = collect(1:nbr_of_op)
    
    while !isempty(remaining_operation)
        # println("scheduler_process is processing ....")
        # at each iteration the process check if the remaining operation are ready to be processed
        
        #here we decide how long this process should sleep
        current_time = now(env)
        #first_up_comming_start_time = minimum(operations_df.start_time)
        #first_up_coming_available_time = minimum(operations_df.available_time)
        sleep_duration = minimum(max.(operations_df.available_time[remaining_operation], operations_df.start_time[remaining_operation])) - current_time
        # sleep_duration = min(first_up_coming_available_time, first_up_comming_start_time) - current_time
        
        #@yield timeout(env, 1)
        if sleep_duration > 0 
            #println("scheduler_process sleep")
            try @yield timeout(env , sleep_duration) catch end 
            # also if new available time is set so this process will wake up too to check
            # if there is any operation can be added to their queue
            #println("scheduler_process wake up")
        end
        for i in remaining_operation
            current_time = now(env)
            if operations_df.available_time[i] <= current_time && operations_df.start_time[i] <= current_time
                correspond_machine = operations_df.machine[i]
                # here we add this op to its corresponds machine's queue
                #println("add operation $i to the machine $correspond_machine's queue at $current_time") 
                push!(machines_df.queue[correspond_machine], i)
                # delete this operation from the remaing machine_operation_list
                filter!(x-> x != i, remaining_operation)
                remaining_operation
                # notify the correspondant machine if it is sleeping
                if ! machines_df.state[correspond_machine] #check if the machine whether is sleep or not
                    try @yield interrupt(machines_df.machine_proc[correspond_machine]) catch end 
                end
            end
        end
    end 
    finish = true
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


function calculate_operations_holding_cost!(node_id::Integer)
    h = operations_df.duration[node_id] * 0.01
    for i in outneighbors(product_graph, node_id)
       h += calculate_operations_holding_cost!(i)
    end
    operations_df.holding_cost[node_id] = h 
end

function MTO_product(decision_variable)

    @assert length(decision_variable) == nbr_of_op + nbr_of_machines "decision variable must be on the size of the problem..."
    start_time = decision_variable[1:nbr_of_op]
    priority_rule = priority_rule_list[Int.(decision_variable[nbr_of_op+1:end])]
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
                                   priority_rule = priority_rule,
                                   machine_proc = [ @process machine(sim,i) for i in 1:nbr_of_machines ] )
      
    global scheduler_proc = @process scheduler_process(sim)
    # @process start_time_control(sim)
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
                            (operations_df.available_time[inneighbors(product_graph, i.id_op)[1]] - i.completion_time)
                            for i in  eachrow( filter(row -> !isempty(inneighbors(product_graph, row.id_op)) , operations_df) ) )
    #here we assume the product has only one final node and always is the first node 
    earliness_cost = operations_df.holding_cost[1] * max(due_date - operations_df.completion_time[1], 0)
    tardiness_cost = 2 * operations_df.holding_cost[1] * max(operations_df.completion_time[1] - due_date , 0)
    return WIP_holding_cost + earliness_cost + tardiness_cost
end
MTO_product([37,82,92,71,97,58,62,12,6,2,6])
