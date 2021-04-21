include("../../../simulations/GG1K_simulation.jl")
using distances

struct ParticleSwarm{T} 
    lower::Vector{T}
    upper::Vector{T}
    n_particles::Int
end

ParticleSwarm(; lower = [], upper = [], n_particles = 0) = ParticleSwarm(lower, upper, n_particles)

mutable struct ParticleSwarmState{Tx,T} 
    x::Tx
    f_x
    iteration::Int
    lower::Tx
    upper::Tx
    c1::T # Weight variable; currently not exposed to users
    c2::T # Weight variable; currently not exposed to users
    w::T  # Weight variable; currently not exposed to users
    limit_search_space::Bool
    n_particles::Int
    X #position of particles
    V # velecity of particles
    X_best# best posistion for each particle
    score::Vector # current score for each particle
    best_score::Vector #best score for each particle
    x_learn
    current_state
    iterations::Int
end

function initial_state(method::ParticleSwarm, options, objfun, initial_x::AbstractArray{T}) where T
    
    n = length(initial_x)

    @assert length(method.lower) == length(method.upper) "lower and upper must be of same length."
    if length(method.lower) > 0
        lower = copyto!(similar(initial_x), copy(method.lower))
        upper = copyto!(similar(initial_x), copy(method.upper))
        limit_search_space = true
        @assert length(lower) == length(initial_x) "limits must be of same length as x_initial."
        @assert all(upper .> lower) "upper must be greater than lower"
    else
        lower = copy(initial_x)
        upper = copy(initial_x)
        limit_search_space = false
    end

    if method.n_particles > 0
        if method.n_particles < 3
          @warn("Number of particles is set to 3 (minimum required)")
          n_particles = 3
        else
          n_particles = method.n_particles
        end
    else
      # user did not define number of particles
       n_particles = maximum([3, length(initial_x)])
    end
    c1 = T(2)
    c2 = T(2)
    w = T(1)

    X = Array{T,2}(undef, n, n_particles)
    V = Array{Float64,2}(undef, n, n_particles)
    X_best = Array{T,2}(undef, n, n_particles)
    dx = zeros(T, n)
    score = zeros( n_particles)
    x = copy(initial_x)
    best_score = zeros( n_particles)
    x_learn = copy(initial_x)

    current_state = 0

    score[1] = objfun(initial_x)

    # if search space is limited, spread the initial population
    # uniformly over the whole search space
    if limit_search_space
        for i in 1:n_particles
            for j in 1:n
                ww = upper[j] - lower[j]
                X[j, i] = round(lower[j] + ww * rand())
                X_best[j, i] = X[j, i]
                V[j, i] = round(ww * (rand() * T(2) - T(1)) / 10)
            end
        end
    else
        for i in 1:n_particles
            for j in 1:n
                if i == 1
                    if abs(initial_x[i]) > T(0)
                        dx[j] = abs(initial_x[i])
                    else
                        dx[j] = T(1)
                    end
                end
                X[j, i] = round(initial_x[j] + dx[j] * rand())
                X_best[j, i] = X[j, i]
                V[j, i] = round(abs(X[j, i]) * (rand() * T(2) - T(1)))
            end
        end
    end

    for j in 1:n
        X[j, 1] = initial_x[j]
        X_best[j, 1] = initial_x[j]
    end

    for i in 2:n_particles
        score[i] = objfun(X[:, i])
    end

    ParticleSwarmState(
        x,
        score[1],
        0,
        lower,
        upper,
        c1,
        c2,
        w,
        limit_search_space,
        n_particles,
        X,
        V,
        X_best,
        score,
        best_score,
        x_learn,
        0,
        1000)
end
 lower=[1,1,1]
 upper=[5,5,5]
initial_x
 dim=3
 f=sim_GG1K

function housekeeping!(score, best_score, X, X_best, best_point,
                       F, n_particles)
    n = size(X, 1)
    for i in 1:n_particles
        if score[i] <= best_score[i]
            best_score[i] = score[i]
            for k in 1:n
                X_best[k, i] = X[k, i]
            end
            if score[i] <= F
                for k in 1:n
                  	best_point[k] = X[k, i]
                end
              	F = score[i]
            end
        end
    end
    return F
end

function get_mu_1(f::Tx) where Tx
    if Tx(0) <= f <= Tx(4)/10
        return Tx(0)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return Tx(5) * f - Tx(2)
    elseif Tx(6)/10 < f <= Tx(7)/10
        return Tx(1)
    elseif Tx(7)/10 < f <= Tx(8)/10
        return -Tx(10) * f + Tx(8)
    else
        return Tx(0)
    end
end

function get_mu_2(f::Tx) where Tx
    if Tx(0) <= f <= Tx(2)/10
        return Tx(0)
    elseif Tx(2)/10 < f <= Tx(3)/10
        return Tx(10) * f - Tx(2)
    elseif Tx(3)/10 < f <= Tx(4)/10
        return Tx(1)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return -Tx(5) * f + Tx(3)
    else
        return Tx(0)
    end
end

function get_mu_3(f::Tx) where Tx
    if Tx(0) <= f <= Tx(1)/10
        return Tx(1)
    elseif Tx(1)/10 < f <= Tx(3)/10
        return -Tx(5) * f + Tx(3)/2
    else
        return Tx(0)
    end
end

function get_mu_4(f::Tx) where Tx
    if Tx(0) <= f <= Tx(7)/10
        return Tx(0)
    elseif Tx(7)/10 < f <= Tx(9)/10
        return Tx(5) * f - Tx(7)/2
    else
        return Tx(1)
    end
end
function get_swarm_state(X::AbstractArray{Tx}, score, best_point, previous_state) where Tx
    # swarm can be in 4 different states, depending on which
    # the weighing factors c1 and c2 are adapted.
    # New state is not only depending on the current swarm state,
    # but also from the previous state.
    n, n_particles = size(X)
    f_best, i_best = findmin(score)
    d = zeros(n_particles)
    for i in 1:n_particles
        dd = Tx(0.0)
        for k in 1:n_particles
            for dim in 1:n
                @inbounds ddd = (X[dim, i] - X[dim, k])
                dd += ddd * ddd
            end
        end
        d[i] = sqrt(dd)
    end
    dg = d[i_best]
    dmin = Base.minimum(d)
    dmax = Base.maximum(d)

    f = (dg - dmin) / max(dmax - dmin, sqrt(eps(Float64)))

    mu = zeros(4)
    mu[1] = get_mu_1(f)
    mu[2] = get_mu_2(f)
    mu[3] = get_mu_3(f)
    mu[4] = get_mu_4(f)
    best_mu, i_best_mu = findmax(mu)
    current_state = 0

    if previous_state == 0
        current_state = i_best_mu
    elseif previous_state == 1
        if mu[1] > 0
            current_state = 1
        else
          if mu[2] > 0
              current_state = 2
          elseif mu[4] > 0
              current_state = 4
          else
              current_state = 3
          end
        end
    elseif previous_state == 2
        if mu[2] > 0
            current_state = 2
        else
          if mu[3] > 0
              current_state = 3
          elseif mu[1] > 0
              current_state = 1
          else
              current_state = 4
          end
        end
    elseif previous_state == 3
        if mu[3] > 0
            current_state = 3
        else
          if mu[4] > 0
              current_state = 4
          elseif mu[2] > 0
              current_state = 2
          else
              current_state = 1
          end
        end
    elseif previous_state == 4
        if mu[4] > 0
            current_state = 4
        else
            if mu[1] > 0
                current_state = 1
            elseif mu[2] > 0
                current_state = 2
            else
                current_state = 3
            end
        end
    end
    return current_state, f
end

function update_swarm_params!(c1, c2, w, current_state, f::T) where T

    delta_c1 = T(5)/100 + rand(T) / T(20)
    delta_c2 = T(5)/100 + rand(T) / T(20)

    if current_state == 1
        c1 += delta_c1
        c2 -= delta_c2
    elseif current_state == 2
        c1 += delta_c1 / 2
        c2 -= delta_c2 / 2
    elseif current_state == 3
        c1 += delta_c1 / 2
        c2 += delta_c2 / 2
    elseif current_state == 4
        c1 -= delta_c1
        c2 -= delta_c2
    end

    if c1 < T(3)/2
        c1 = T(3)/2
    elseif c1 > T(5)/2
        c1 = T(5)/2
    end

    if c2 < T(3)/2
        c2 = T(5)/2
    elseif c2 > T(5)/2
        c2 = T(5)/2
    end

    if c1 + c2 > T(4)
        c_total = c1 + c2
        c1 = c1 / c_total * 4
        c2 = c2 / c_total * 4
    end

    w = 1 / (1 + T(3)/2 * exp(-T(26)/10 * f))
    return w, c1, c2
end

function update_swarm!(X::AbstractArray{Tx}, X_best, best_point, n, n_particles, V,
                       w, c1, c2) where Tx
  # compute new positions for the swarm particles
  for i in 1:n_particles
      for j in 1:n
          r1 = rand()
          r2 = rand()
          vx = X_best[j, i] - X[j, i]
          vg = best_point[j] - X[j, i]
          V[j, i] = V[j, i]*w + c1*r1*vx + c2*r2*vg
          X[j, i] = round(X[j, i] + V[j, i])
      end
    end
end

function limit_X!(X, lower, upper, n_particles, n)
    # limit X values to boundaries
    for i in 1:n_particles
        for j in 1:n
            if X[j, i] < lower[j]
              	X[j, i] = lower[j]
            elseif X[j, i] > upper[j]
              	X[j, i] = upper[j]
            end
        end
    end
    nothing
end

function compute_cost!(f,
                       n_particles::Int,
                       X::Matrix,
                       score::Vector)

    for i in 1:n_particles
        score[i] = f(X[:, i])
    end
    nothing
end
function ParticleSwarmAlgo(f, lower::Array{Int,1}, upper::Array{Int,1} ,dim)
    # initialisation
    initial_x=Array{Int,1}(undef,dim)
    for i in 1:dim
        initial_x[i]=lower[i]+ round(rand()*(upper[i]-lower[i]))
    end
    method= ParticleSwarm(lower= lower, upper = upper, n_particles= dim +1)
    
    state= initial_state(method,[],f,initial_x)
    
    n = dim
   
    while state.iteration<1000
        state.f_x = housekeeping!(state.score,
                                state.best_score,
                                state.X,
                                state.X_best,
                                state.x,
                                state.f_x,
                                state.n_particles)
        # Elitist Learning:
        # find a new solution named 'x_learn' which is the current best
        # solution with one randomly picked variable being modified.
        # Replace the current worst solution in X with x_learn
        # if x_learn presents the new best solution.
        # In all other cases discard x_learn.
        # This helps jumping out of local minima.
        worst_score, i_worst = findmax(state.score)
        
        state.x_learn=state.x
        
        random_index = rand(1:n)
        random_value = randn()
        sigma_learn = 1 - (1 - 0.1) * state.iteration / state.iterations

        r3 = randn() * sigma_learn

        if state.limit_search_space
            state.x_learn[random_index] = round(state.x_learn[random_index] + (state.upper[random_index] - state.lower[random_index]) / 3.0 * r3)
            if state.x_learn[random_index] < state.lower[random_index]
                state.x_learn[random_index] = state.lower[random_index]
            elseif state.x_learn[random_index] > state.upper[random_index]
                state.x_learn[random_index] = state.upper[random_index]
            end
        else
            state.x_learn[random_index] = round(state.x_learn[random_index] + state.x_learn[random_index] * r3)
        end
        
        score_learn = f(state.x_learn)
        state.X_best[:, i_worst]=state.x_learn
        if score_learn < state.f_x
            state.f_x = score_learn 
            state.X_best[:, i_worst] = state.x_learn
            state.X[:, i_worst] = state.x_learn
            state.x[:] = state.x_learn
            state.score[i_worst] = score_learn
            state.best_score[i_worst] = score_learn
        end

        # TODO find a better name for _f (look inthe paper, it might be called f there)
        state.current_state, _f = get_swarm_state(state.X, state.score, state.x, state.current_state)
        #state.w, state.c1, state.c2 = update_swarm_params!(state.c1, state.c2, state.w, state.current_state, _f)
        update_swarm!(state.X, state.X_best, state.x, n, state.n_particles, state.V, state.w, state.c1, state.c2)

        if state.limit_search_space
            limit_X!(state.X, state.lower, state.upper, state.n_particles, n)
        end
        compute_cost!(f, state.n_particles, state.X, state.score)

        state.iteration += 1
        println(state.x)
    end
    state
end

state= ParticleSwarmAlgo(sim_GG1K,[1,1,1],[5,5,5],3)
