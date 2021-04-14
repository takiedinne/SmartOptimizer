include("../../../simulations/GG1K_simulation.jl")
using LinearAlgebra

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

struct NelderMead{Ts <: Simplexer, Tp <: NMParameters} 
    initial_simplex::Ts
    parameters::Tp
end

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
    rmul!(c, T(1)/n)
end

centroid(simplex, h) = centroid!(Float64.(similar(simplex[1])), simplex, h)
nmobjective(y::Vector, m::Integer, n::Integer) = sqrt(var(y) * (m / n))

mutable struct NelderMeadState{Tx, T, Tfs} 
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

function initial_state(method::NelderMead, options, objfun::Function, initial_x)
    T = eltype(method.parameters.α)
    n = length(initial_x)
    m = n + 1
    simplex = simplexer(method.initial_simplex, initial_x)
    f_simplex = zeros(m)

    for i in 1:length(simplex)
        f_simplex[i] = objfun(simplex[i])
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
          "initial")
end


function sgnd(k)
    result=1
    if k==0 result=0 end
    if k<0 result = -1 end 
    result
end
function NelderMeadAlgo(objfun::F, bounds, dim) where {F}
    # initialisation
    method= NelderMead(AffineSimplexer(), FixedParameters())
    initial_x=rand(bounds[1]:bounds[2],dim)
    state= initial_state(method,[],objfun,initial_x)
    n, m = length(state.x), state.m
    while state.iteration<1000 && state.nm_x > 0.1
        # Augment the iteration counter
        shrink = false
        n, m = length(state.x), state.m
        centroid!(state.x_centroid, state.simplex, state.i_order[m])# doesn't need bounds check neccesarily the centroid is ouside the bounds
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
            if state.x_reflect[j] > bounds[2]
                state.x_reflect[j] = bounds[2]
            end
            if state.x_reflect[j] < bounds[1]
                state.x_reflect[j] = bounds[1]
            end
        end

        
        f_reflect = objfun(state.x_reflect)
        if f_reflect < state.f_lowest
            # Compute an expansion
            @inbounds for j in 1:n
                state.x_cache[j] = state.x_highest[j] + state.β *
                                (floor(abs(state.x_highest[j]-state.x_centroid[j]))+1) * 
                                sgnd(state.x_centroid[j]-state.x_highest[j])
                if state.x_cache[j] > bounds[2]
                    state.x_cache[j] = bounds[2]
                end
                if state.x_cache[j] < bounds[1]
                    state.x_cache[j] = bounds[1]
                end
            end
            
            f_expand =objfun(state.x_cache)

            if f_expand < f_reflect
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_expand
                state.step_type = "expansion"
            else
                copyto!(state.simplex[state.i_order[m]], state.x_reflect)
                @inbounds state.f_simplex[state.i_order[m]] = f_reflect
                state.step_type = "reflection"
            end
            # shift all order indeces, and wrap the last one around to the first
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
                    if state.x_cache[j] > bounds[2]
                        state.x_cache[j] = bounds[2]
                    end
                    if state.x_cache[j] < bounds[1]
                        state.x_cache[j] = bounds[1]
                    end   
                end
                state.x_cache
                state.x_reflect
                state.x_centroid
                f_outside_contraction = objfun(state.x_cache)
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
                    if state.x_cache[j] > bounds[2]
                        state.x_cache[j] = bounds[2]
                    end
                    if state.x_cache[j] < bounds[1]
                        state.x_cache[j] = bounds[1]
                    end
                end
                f_inside_contraction = objfun(state.x_cache)
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
                state.f_simplex[ord] = objfun(state.simplex[ord])
            end
            step_type = "shrink"
            sortperm!(state.i_order, state.f_simplex)
        end
        state.nm_x = nmobjective(state.f_simplex, n, m)
        state.iteration+=1
    end
    copyto!(state.x_lowest, state.simplex[state.i_order[1]])
    copyto!(state.x_second_highest, state.simplex[state.i_order[n]])
    copyto!(state.x_highest, state.simplex[state.i_order[m]])
    state.f_lowest = state.f_simplex[state.i_order[1]]
    f_second_highest = state.f_simplex[state.i_order[n]]
    f_highest = state.f_simplex[state.i_order[m]]
    state
end

final_state= NelderMeadAlgo(sim_GG1K,(1,5),3)
final_state.x_lowest
final_state.f_lowest
final_state.simplex