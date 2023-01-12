#=using CSV
using DataFrames
using Evolutionary
using Random
data = CSV.read("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\SmartOptimizer\\simulations\\BrainFuck\\data.csv", DataFrame)

global FG, LL, BD, CG

function f_obj(code::Expr)
    results = []
    for i in 1:nrow(data)
        global FG, CG, LL, BD = data.FG[i], data.CG[i], data.LL[i], data.BD[i]
        try 
            f = eval(code)
            #println("n° : $i  f= $f")
            push!(results, f)
        catch 
            results = Inf # penality
            break
        end
    end
    if results != Inf
        RMSE = sqrt(sum((results[i] - data.ASR[i])^2 for i in 1:nrow(data) ) / nrow(data) )
    else
        RMSE = Inf
    end
end
# code = Meta.parse("exp(BD - 1 / CG) + √ log(LL) + exp(BD) + √(LL + BD) + cos(1 / (sin(sin(-1 * ( FG + BD))))) + exp(BD)")
# code = Meta.parse("8 * BD^2 + LL * (FG^2 - 100 * FG) * (8 * BD^2 - 25) / 250000 ")

# f_obj(code)
fun_set = [:+, :-, :*, :/, :√, :sin, :cos]
anary_list = [2, 2, 2, 2, 1, 1, 1, 1, 1]
terminal_set = [1, :FG, :CG, :LL, :BD]
symbol_set = [] 
append!(symbol_set, fun_set)
append!(symbol_set, terminal_set)


# these random generation  code need as input the list of symbols and the level
function grow_random_code(symbol_list, level::Int, max_depth::Int)
    
    if level <= max_depth
        i_next_symbol = rand(1:length(symbol_list))
        if symbol_list[i_next_symbol] in fun_set
            tmp_exp = Expr(:call) 
            func_symbol = symbol_list[i_next_symbol]
            push!(tmp_exp.args, func_symbol)
            level += 1
            symbol_list = (level == max_depth) ? terminal_set : symbol_set
            
            for _ in 1:anary_list[i_next_symbol]
                generated_exp = grow_random_code(symbol_list, level, max_depth)
                push!(tmp_exp.args, generated_exp)
            end
            return tmp_exp
        else # the selected element is a terminal
            symbol_list[i_next_symbol]
            return symbol_list[i_next_symbol]
        end
    end
end

function full_random_code(symbol_list, level::Int, max_depth::Int)
    
    if level <= max_depth
        i_next_symbol = rand(1:length(symbol_list))
        if symbol_list[i_next_symbol] in fun_set
            tmp_exp = Expr(:call) 
            func_symbol = symbol_list[i_next_symbol]
            push!(tmp_exp.args, func_symbol)
            level += 1
            symbol_list = (level == max_depth) ? terminal_set : fun_set
            
            for _ in 1:anary_list[i_next_symbol]
                generated_exp = full_random_code(symbol_list, level, max_depth)
                push!(tmp_exp.args, generated_exp)
            end
            return tmp_exp
        else # the selected element is a terminal
            symbol_list[i_next_symbol]
            return symbol_list[i_next_symbol]
        end
    end
end

# functions for generating the population
function ramped_half_and_half(max_depth, population_size)
    # generate equal nbr of individual of depth i where i ∈ 2:max_depth
    # and the sum of all the individal will be population size 
    @assert max_depth >= 2  "max_depth must be gratter or equal 2 ... "
    individual_nbr_for_each_depth = floor(Int, population_size / (max_depth - 1))
    missing_number = population_size % (max_depth - 1)
    full_poulation_size = floor(Int, individual_nbr_for_each_depth / 2)
    grow_population_size = isodd(individual_nbr_for_each_depth) ? full_poulation_size+1 : full_poulation_size
    population = []
    for i in 2:max_depth
        for _ in 1:grow_population_size
            push!(population, grow_random_code(fun_set, 0, i))
        end
        for _ in 1:full_poulation_size
            push!(population, full_random_code(fun_set, 0, i))
        end
    end
    # fill the missing number
    for i in 1:missing_number
        method = rand([grow_random_code, full_random_code])
        random_max_depth = rand(2:max_depth)
        push!(population, method(fun_set, 0, random_max_depth))
    end
    population
end

function indv_length(indv::T) where T
    if T == Symbol
        return 1 
    end
    stack = []
    append!(stack, indv.args)
    l = 0
    while !isempty(stack)
        current = popfirst!(stack)
        if current isa (Expr)
            for arg in current.args
                push!(stack, arg)
            end
        else # not a expression
            l += 1
        end
    end
    l
end

function get_fragment(indv::T, selected_point_indv) where T
    if T == Symbol
        return indv, -1
    end
    stack = []
    current, father, current_order = indv, indv, -1
    browsed_element_counter = 1
    append!(stack, [(indv, i, indv.args[i]) for i in 1:length(indv.args)])
    while !isempty(stack) && browsed_element_counter < selected_point_indv
        father, current_order, current = popfirst!(stack)
        if current isa(Expr)
            for i in 1:length(current.args)
                push!(stack, (current, i, current.args[i]))
            end
            browsed_element_counter += 1
        elseif current ∈ terminal_set # not a expression
            browsed_element_counter += 1
        end 
    end
    father, current_order
end

function tree_crossover(indv1::T1, indv2::T2) where {T1, T2}
    length_indv1 = indv_length(indv1)
    length_indv2 = indv_length(indv2)

    sibling1 = (T1 == Expr) ? copy(indv1) : indv1
    sibling2 = (T2 == Expr) ? copy(indv2) : indv2
    crossover_point_indv1 = rand(1:length_indv1)
    crossover_point_indv2 = rand(1:length_indv2)
    # browse the individual to select the fragment cross over
    father1, order1 = get_fragment(sibling1, crossover_point_indv1)
    father2, order2 = get_fragment(sibling2, crossover_point_indv2)
    # exchange the two fragments
    fragment1 = (order1 == -1) ? father1 : father1.args[order1]
    fragment2 = (order2 == -1) ? father2 : father2.args[order2]
    if order1 == -1
        sibling1 = fragment2
    else
        father1.args[order1] = fragment2
    end
    if order2 == -1
        sibling2 = fragment1
    else
        father2.args[order2] = fragment1
    end
    # check if the new siblings are symbols so we convert them to Expr
        sibling1, sibling2
end

function tree_mutation(indv::T, max_depth) where T
    sibling = isa(indv, Expr) ? copy(indv) : indv
    if T==Expr
        mutation_point_indv = rand(1:indv_length(sibling))
        father, order = get_fragment(sibling, mutation_point_indv)
    else
        order = -1
    end
    method = rand([grow_random_code, full_random_code])
    if order == -1
        sibling = method(symbol_set, 0, max_depth)
    else
        father.args[order] =  method(symbol_set, 0, max_depth)
    end
    sibling
end

function tree_edition(indv::Expr)
    # this function check if there is some edition rule to enhance the indv
    # check if sub tree return 0 so we delete it or we remplace by their correspondant value
    stack = []
    current, father, current_order = indv, indv, -1
    
    append!(stack, [(indv, i, indv.args[i]) for i in 1:length(indv.args)])
    while !isempty(stack)
        father, current_order, current = popfirst!(stack)
        if current isa(Expr)
            
            for i in 1:length(current.args)
                push!(stack, (current, i, current.args[i]))
            end
            browsed_element_counter += 1
        elseif current ∈ terminal_set # not a expression
            browsed_element_counter += 1
        end 
    end
    father, current_order
end

function tree_genetic_programming(max_depth, population_size, nbr_of_generation, f_obj;
                                     crossoverRate=0.9, mutationRate=0.1)
    pop = ramped_half_and_half(max_depth, population_size)
    fit_pop = [f_obj(indv) for indv in pop ]
    current_generation = 1 
    
    minfit, fitidx = findmin(fit_pop)
    
    best_code = pop[fitidx]
    best_fit = fit_pop[fitidx]
   
    while current_generation <= nbr_of_generation
        # constructing the new population
        new_pop = similar(pop)
         # Select offspring
         # as we are in minimization problem 
        selected = rouletteinv(fit_pop, population_size)
        
        # Perform mating
        offidx = randperm(population_size) #get random ordre of population
        #perform crossover 
        for i in 1:2:population_size
            j = (i == population_size) ? i-1 : i+1
            if rand() < crossoverRate
                new_pop[i], new_pop[j] = tree_crossover(pop[selected[offidx[i]]], pop[selected[offidx[j]]])
            else
                new_pop[i], new_pop[j] = pop[selected[i]], pop[selected[j]]
            end
        end
        # Perform mutation
        for i in 1:population_size
            if rand() < mutationRate
                tree_mutation(new_pop[i], max_depth)
            end
        end
        pop = new_pop
        fit_pop = Float64[]
        for indv in pop
            if isa(indv, Symbol) || isa(indv, Integer)
                indv = Expr(Symbol(indv))
            end
            push!(fit_pop, f_obj(indv))
        end
        # find the best individual
        minfit, fitidx = findmin(fit_pop)
        if minfit <= best_fit
            best_code = pop[fitidx]
            best_fit = fit_pop[fitidx]
        end
        println("-----------------------------------------------------")
        println(" current generation: $current_generation")
        println("best fitness: $best_fit")
        println("-----------------------------------------------------")
        
        current_generation += 1
    end # end while
    best_code, best_fit 
end
r = tree_genetic_programming(6, 100, 100, f_obj)
=#
using Evolutionary
using Random
using Plots
using CSV
using DataFrames
data = CSV.read("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\SmartOptimizer\\simulations\\BrainFuck\\data.csv", DataFrame)

Plots.gr()
default(fmt = :png)

Random.seed!(42);
xs = [ data[i, [:FG, :CG, :LL, :BD]] |> Array for i in 1:nrow(data)]
#xs = range(-20, 20, length=135) |> collect

ys = data[!, :ASR] |> Array
xs, ys

fitobj(expr) = try 
                    sum( abs2.(ys - Evolutionary.Expression(expr).(xs)) ) |> sqrt
                catch 
                    Inf
                end


expr = Expr(:call, *, (:call, cos, :x), Expr(:call, -, :w, :z))
x = Inf
println("Obj. func = ", fitobj(expr))

syms = Dict(:FG=>1, :CG=>1, :LL=>1, :BD=>1, (rand) => 1) 

funcs = Dict((+) => 2, (-) => 2,  (*) => 2, (/) => 2, (cos) => 1, (sin) => 1)

#Random.seed!(987498737423);
res = Evolutionary.optimize(fitobj,
    TreeGP(
        populationSize = 500,
        terminals = syms,
        functions = funcs,
        mindepth = 1,
        maxdepth = 3,
        initialization = :grow,
        simplify = Evolutionary.simplify!,
        optimizer = GA(
            selection = uniformranking(2),
            mutationRate = 0.1,
            crossoverRate = 0.95,
            ɛ = 0.001
        ),
    ),
    Evolutionary.Options(iterations=500, show_trace=true, show_every=10)
)

ex = Evolutionary.Expression(Evolutionary.minimizer(res))

X = hcat(ones(n), xs)
β = inv(X'X)*X'ys

scatter(xs, ys, label="Data", legend=:topleft)
plot!(xs, β[2].*(xs).+β[1], label="Linear")
plot!(xs, ex.(xs), label="Symbolic")

abs(rand())