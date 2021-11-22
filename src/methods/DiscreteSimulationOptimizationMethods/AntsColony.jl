import StatsBase: Weights, sample   
"""
ants colony optimization is first presented on [1], and the implementation is taken from [2]
with majour modification.
in this implementation we assume that each variable represent a stage for the search spacce
for example if we have a problem of 2 variables so we have 3 stage (we choose the value of the first
variable and then for the second variable and so one) we assume each value (represented by a node)
can move to all the possible values for the next variable e.g., proble of 2 variable wich take 2 and 3  values 
respactively so we have a graphe of initial node which is connected to 2 nodes and each node from this stage is connected to the 3 other nodes
it is worth to mention that we don't have any priority between the nodes because we are in a black box context                            
[1]: M. Dorigo, G. Di Caro, and L. M. Gambardella, “Ant algorithms for discrete optimization,” Artif. Life, 
    vol. 5, no. 2, pp. 137–172, 1999, doi: 10.1162/106454699568728.
[2]: M. J. Kochenderfer and T. A. Wheeler, Algorithms for optimization. 2014.
"""
### important it is necessary that the search sace is discrete and finite
struct AntColonySearcher <: LowLevelHeuristic
    method_name::String
    ants_nbr::Integer
   
    #hyper parameter
    α # between 0 and 1 represent the impact of phermone level
    β # between 0 and 1 the imact of the priority
    ρ # evaporation rate

    AntColonySearcher(; ants_nbr=10, alpha=0.5, beta=0.5, rho= 0.5) = new("Ants colony method",
                                                                        ants_nbr, alpha, beta,rho)
end


mutable  struct AntColonySearcherState{T} <:State
    x::AbstractArray{T,1} # here x is the same x_current
    f_x
    G # the graph
    τ # phermone level for each arc
    η # represent the priority of each arc
    attrac # the attractiveness for each arc
end 

function constructGraph(upper, lower)
    nbr_vertex = sum(upper[i] - lower[i] + 1 for i in 1:length(upper)) + 1
    g = SimpleDiGraph(nbr_vertex)
    for j in 1:(upper[1] - lower[1] + 1)
        add_edge!(g, 1, j+1)
    end
    currentVertexIndex = 2
    k = 1
    for i in 1:length(upper)-1
        k += sum(upper[i] - lower[i] + 1)
        for j in 1:(upper[i] - lower[i] + 1)
            for h in 1:(upper[i+1] - lower[i+1] + 1)
                goalVertex = k + h
                add_edge!(g, currentVertexIndex, goalVertex)
            end
            currentVertexIndex += 1
        end 
    end
    g
end

function edge_attractiveness(graph, τ, η; α=1, β=5)
    A = Dict()
    for i in 1 : nv(graph)
        neighbors = outneighbors(graph, i)
        for j in neighbors
            if τ[(i,j)]<0 println("here...")  end 
            v = τ[(i,j)]^α * η[(i,j)]^β
            A[(i,j)] = v
        end
    end
    return A
end
function initial_state(method::AntColonySearcher, problem::Problem{T}) where {T<:Number}
    @assert (length(problem.upper) == length(problem.lower) > 0) " the problem must be finit"
    x = problem.x_initial
    fit = problem.objective(x)
    # initialise the phermone level
    G = constructGraph(problem.upper, problem.lower)
    τ = Dict((e.src,e.dst)=>1.0 for e in edges(G))
    η = Dict((e.src,e.dst)=>1.0 for e in edges(G))
    attrac = Dict((e.src,e.dst)=>1.0 for e in edges(G))
    AntColonySearcherState(x, fit, G, τ, η, attrac)
end

function run_ant(G, τ, A, problem)
    upper, lower = problem.upper, problem.lower
    #construct the solution
    x = [1]
    neighbors = outneighbors(G, 1)
    while !isempty(neighbors)
        i = x[end]
        as = [A[(i,j)] for j in neighbors]
        push!(x, neighbors[sample(Weights(as))])
        
        neighbors = outneighbors(G, x[end])
    end
    # translate the solution
    x_translated = []
    k = 2
    for i in 2:length(x)
        push!(x_translated, lower[i-1] + x[i] - k)  
        k += sum(upper[i-1] - lower[i-1] + 1)
    end

    l = problem.objective(x_translated)
    for i in 2 : length(x)
        τ[(x[i-1],x[i])] += 1/l # you must see another phermone update function which take on consideration
                                # the fact where we have negative fitness values
    end
    return (x_translated, l)
end
function update_state!(method::AntColonySearcher, problem::Problem{T}, iteration::Int, state::AntColonySearcherState) where {T}
    nbrSim = 0
    state.attrac = edge_attractiveness(state.G, state.τ, state.η, α=method.α, β= method.β)
    #evaporation
    for (e,v) in state.τ
       state.τ[e] = (1-method.ρ) * v
    end
    #run ants
    for ant in 1 : method.ants_nbr
        x, fit = run_ant(state.G,state.τ, state.attrac, problem)
        nbrSim += 1
        if fit <= state.f_x
            state.x = x
            state.f_x = fit
        end
    end
    state.x, state.f_x, nbrSim
end

function create_state_for_HH(method::AntColonySearcher, problem::Problem{T}, HHState::HH_State) where {T<:Number}
    
end
