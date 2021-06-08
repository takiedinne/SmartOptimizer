abstract type Simplexer end

struct AffineSimplexer <: Simplexer
    a::Float64
    b::Float64
end

AffineSimplexer(;a = 0.025, b = 0.5) = AffineSimplexer(a, b)

function simplexer(S::AffineSimplexer, initial_x::Tx) where Tx
    n = length(initial_x)
    initial_simplex = Tx[copy(initial_x) for i = 1:n+1]
    for j = 1:n
        # here I use round function to ensure that the points are in integer search space
        initial_simplex[j+1][j] = round((1+S.b) * initial_simplex[j+1][j] + S.a)
    end
    initial_simplex
end

abstract type NMParameters end

struct AdaptiveParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

AdaptiveParameters(;  α = 1.0, β = 1.0, γ = 0.75 , δ = 1.0) = AdaptiveParameters(α, β, γ, δ)
parameters(P::AdaptiveParameters, n::Integer) = (P.α, P.β + 2/n, P.γ - 1/2n, P.δ - 1/n)

struct FixedParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
end

FixedParameters(; α = 2.0, β = 3.0, γ = 0.5, δ = 0.5) = FixedParameters(α, β, γ, δ)
parameters(P::FixedParameters, n::Integer) = (P.α, P.β, P.γ, P.δ)

struct NelderMead{Ts <: Simplexer, Tp <: NMParameters} <: LowLevelHeuristic
    method_name::String
    initial_simplex::Ts
    parameters::Tp
end

NelderMead(;initial_simplex=AffineSimplexer(), parameters=FixedParameters())= NelderMead("Nelder Mead",
                                                                            initial_simplex, parameters)

# centroid except h-th vertex
function centroid!(c::AbstractArray{T}, simplex, h=0) where T
    n = length(c)
    fill!(c, zero(T))
    @inbounds for i in 1:n+1
        if i != h
            xi = simplex[i]
            c .+= xi
        end
    end
    rmul!(c, 1/n)
end

centroid(simplex, h) = centroid!(Float64.(similar(simplex[1])), simplex, h)
nmobjective(y::Vector, m::Integer, n::Integer) = sqrt(var(y) * (m / n))

mutable struct NelderMeadState{Tx, T, Tfs} <:State
    x::Tx
    iteration::Integer
    m::Int
    simplex::Vector{Tx}
    x_centroid
    x_lowest::Tx
    x_second_highest::Tx
    x_highest::Tx
    x_reflect::Tx
    x_cache::Tx
    f_simplex::Tfs
    nm_x::T
    f_lowest::T
    i_order::Vector{Int}
    α::T
    β::T
    γ::T
    δ::T
    step_type::String
end

function initial_state(method::NelderMead, problem::Problem)
    T = eltype(method.parameters.α)
    n = problem.dimension 
    m = n + 1 # simplex size
    initial_x= problem.x_initial
    simplex = simplexer(method.initial_simplex, initial_x)
    #check in bound the simplex
    map(x->check_in_bounds(problem.upper, problem.lower, x), simplex)
    f_simplex = map(x->problem.objective(x), simplex)
    
    # Get the indices that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    
    i_order = sortperm(f_simplex)

    α, β, γ, δ = parameters(method.parameters, n)
    
    NelderMeadState(copy(initial_x), # Variable to hold final minimizer value for MultivariateOptimizationResults
          0,#iteration
          m, # Number of vertices in the simplex
          simplex, # Maintain simplex in state.simplex
          centroid(simplex,  i_order[m]), # Maintain centroid in state.centroid
          copy(initial_x), # Store cache in state.x_lowest
          copy(initial_x), # Store cache in state.x_second_highest
          copy(initial_x), # Store cache in state.x_highest
          copy(initial_x), # Store cache in state.x_reflect
          copy(initial_x), # Store cache in state.x_cache
          f_simplex, # Store objective values at the vertices in state.f_simplex
          T(nmobjective(f_simplex, n, m)), # Store nmobjective in state.nm_x
          f_simplex[i_order[1]], # Store lowest f in state.f_lowest
          i_order, # Store a vector of rankings of objective values
          T(α),
          T(β),
          T(γ),
          T(δ),
          "initial")
end


function sgnd(k)
    result=1
    if k==0 result=0 end
    if k<0 result = -1 end 
    result
end
function update_state!(method::NelderMead, problem::Problem{T} , iteration::Int, state::NelderMeadState) where {T}
    shrink = false
    n, m = length(state.x), state.m
    nbrSim = 0
    centroid!(state.x_centroid, state.simplex, state.i_order[m])# doesn't need bounds check neccesarily the centroid is onside the bounds
    
    copyto!(state.x_lowest, state.simplex[state.i_order[1]])
    copyto!(state.x_second_highest, state.simplex[state.i_order[n]])
    copyto!(state.x_highest, state.simplex[state.i_order[m]])

    state.f_lowest = state.f_simplex[state.i_order[1]]
    f_second_highest = state.f_simplex[state.i_order[n]]
    f_highest = state.f_simplex[state.i_order[m]]

    # Compute a reflection
    @inbounds for j in 1:n
        state.x_reflect[j] = state.x_highest[j] + 
                                state.α * (floor(abs(state.x_highest[j]-state.x_centroid[j]))+1) * 
                                sgnd(state.x_centroid[j]-state.x_highest[j])
    end
    
    check_in_bounds( problem.upper, problem.lower, state.x_reflect)
    
    f_reflect = problem.objective(state.x_reflect)
    nbrSim += 1
    if f_reflect < state.f_lowest
        # Compute an expansion
        @inbounds for j in 1:n
            state.x_cache[j] = state.x_highest[j] + state.β *
                            (floor(abs(state.x_highest[j]-state.x_centroid[j]))+1) * 
                            sgnd(state.x_centroid[j]-state.x_highest[j])
        end
        
        check_in_bounds(problem.upper, problem.lower, state.x_cache)

        f_expand =problem.objective(state.x_cache)
        nbrSim +=1
        if f_expand < f_reflect
            copyto!(state.simplex[state.i_order[m]], state.x_cache)
            @inbounds state.f_simplex[state.i_order[m]] = f_expand
            state.step_type = "expansion"
        else
            copyto!(state.simplex[state.i_order[m]], state.x_reflect)
            @inbounds state.f_simplex[state.i_order[m]] = f_reflect
            state.step_type = "reflection"
        end
        # shift all order indices, and wrap the last one around to the first
        i_highest = state.i_order[m]
        @inbounds for i = m:-1:2
            state.i_order[i] = state.i_order[i-1]
        end
        state.i_order[1] = i_highest
        
        
    elseif f_reflect < f_second_highest
        copyto!(state.simplex[state.i_order[m]], state.x_reflect)
        @inbounds state.f_simplex[state.i_order[m]] = f_reflect
        state.step_type = "reflection"
        sortperm!(state.i_order, state.f_simplex)
        
    else
        if f_reflect < f_highest
            # Outside contraction
            @inbounds for j in 1:n
                state.x_cache[j] = state.x_reflect[j] + round(state.γ *
                    (floor(abs(state.x_reflect[j]-state.x_centroid[j]))+1) * 
                    sgnd(state.x_centroid[j]-state.x_reflect[j]))
                 
            end

            check_in_bounds(problem.upper, problem.lower, state.x_cache)
            f_outside_contraction = problem.objective(state.x_cache)
            nbrSim += 1
            if f_outside_contraction < f_reflect
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_outside_contraction
                state.step_type = "outside contraction"
                sortperm!(state.i_order, state.f_simplex)
            else
                shrink = true
            end
            
        else # f_reflect > f_highest
            # Inside constraction
            @inbounds for j in 1:n
                state.x_cache[j] = state.x_highest[j] + round(state.γ *
                (floor(abs(state.x_highest[j]-state.x_centroid[j]))+1) * 
                sgnd(state.x_centroid[j]-state.x_highest[j]))            
            end

            check_in_bounds(problem.upper, problem.lower, state.x_cache)

            f_inside_contraction = problem.objective(state.x_cache)
            nbrSim += 1
            if f_inside_contraction < f_highest
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_inside_contraction
                state.step_type = "inside contraction"
                sortperm!(state.i_order, state.f_simplex)
            else
                shrink = true
            end
            
        end
    end

    if shrink
        for i = 2:m
            #here we don't need to check if the new point is in the bounded area 
            ord = state.i_order[i]
            copyto!(state.simplex[ord], state.x_lowest + round.(state.δ*(state.simplex[ord]-state.x_lowest)))
            state.f_simplex[ord] = problem.objective(state.simplex[ord])
            nbrSim += 1
        end
        step_type = "shrink"
        sortperm!(state.i_order, state.f_simplex)
    end
    
    # usefull for convergence we measure the size of the simplex
    state.nm_x = nmobjective(state.f_simplex, n, m)
    state.iteration+=1
    
    copyto!(state.x_lowest, state.simplex[state.i_order[1]])
    copyto!(state.x_second_highest, state.simplex[state.i_order[n]])
    copyto!(state.x_highest, state.simplex[state.i_order[m]])

    state.f_lowest = state.f_simplex[state.i_order[1]]
    f_second_highest = state.f_simplex[state.i_order[n]]
    f_highest = state.f_simplex[state.i_order[m]]

    state.x_lowest, state.f_lowest, nbrSim
end
function create_state_for_HH(method::NelderMead, problem::Problem, archive)
    T = eltype(method.parameters.α)
    n = problem.dimension 
    m = n + 1 # simplex size
    nbrSim = 0
    archiveCopy= copy(archive)
    archiveCopy= sort!(archiveCopy,[:fit])
    initial_x= archiveCopy.x[1]
    if nrow(archive) >= m
        simplex = archiveCopy.x[1:m]
        f_simplex = archiveCopy.fit
    else
        simplex = archiveCopy.x
        f_simplex = archiveCopy.fit
        for i in 1:(m-nrow(archive))
            tmp = copy(problem.x_initial)
            random_x!(tmp, n, upper=problem.upper, lower=problem.lower)
            push!(simplex, tmp)
            push!(f_simplex, problem.objective(tmp))
            nbrSim += 1
        end
    end
    
    # Get the indices that correspond to the ordering of the f values
    # at the vertices. i_order[1] is the index in the simplex of the vertex
    # with the lowest function value, and i_order[end] is the index in the
    # simplex of the vertex with the highest function value
    
    i_order = sortperm(f_simplex)

    α, β, γ, δ = parameters(method.parameters, n)
    
    NelderMeadState(copy(initial_x), # Variable to hold final minimizer value for MultivariateOptimizationResults
          0,#iteration
          m, # Number of vertices in the simplex
          simplex, # Maintain simplex in state.simplex
          centroid(simplex,  i_order[m]), # Maintain centroid in state.centroid
          copy(initial_x), # Store cache in state.x_lowest
          copy(initial_x), # Store cache in state.x_second_highest
          copy(initial_x), # Store cache in state.x_highest
          copy(initial_x), # Store cache in state.x_reflect
          copy(initial_x), # Store cache in state.x_cache
          f_simplex, # Store objective values at the vertices in state.f_simplex
          T(nmobjective(f_simplex, n, m)), # Store nmobjective in state.nm_x
          f_simplex[i_order[1]], # Store lowest f in state.f_lowest
          i_order, # Store a vector of rankings of objective values
          T(α),
          T(β),
          T(γ),
          T(δ),
          "initial"), nbrSim

end
