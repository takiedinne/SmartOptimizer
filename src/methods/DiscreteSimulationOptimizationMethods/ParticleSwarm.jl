const c1 = 2
const c2 = 2
const  w =  1

struct ParticleSwarm <: LowLevelHeuristic
    method_name::String
    n_particles::Int
end

ParticleSwarm(;n_particles = 0) = ParticleSwarm("Particle Swarm", n_particles)

mutable struct ParticleSwarmState{Tx,T} <:State
    x::Tx
    f_x
    iteration::Int
    c1::T # Weight variable; currently not exposed to users
    c2::T # Weight variable; currently not exposed to users
    w::T  # Weight variable; currently not exposed to users
    limit_search_space::Bool
    
    X #position of particles
    V # velecity of particles
    X_best# best posistion for each particle
    score::Vector # current score for each particle
    best_score::Vector #best score for each particle
    x_learn #used to accelerate the search
    current_state 
    iterations::Int
end

function initial_state(method::ParticleSwarm, problem::Problem{T}) where {T<:Number}
    
    n = length(problem.x_initial)
    upper = problem.upper
    lower = problem.lower
    if 0 < method.n_particles < 3 
        @warn("Number of particles is set to 3 (minimum required)")
        method.n_particles = 3
    elseif method.n_particles==0
        # user did not define number of particles
        n_particles = maximum([3, length(problem.x_initial)])
    end

    X = Array{T,2}(undef, n, n_particles)
    V = Array{Float64,2}(undef, n, n_particles)
    X_best = Array{T,2}(undef, n, n_particles)
    dx = zeros(T, n)
    score = zeros( n_particles)
    x = copy(problem.x_initial)
    best_score = zeros(n_particles)
    x_learn = copy(problem.x_initial)

    current_state = 0
    score[1] = problem.objective(problem.x_initial)
    limit_search_space= false
    
    # if search space is limited, spread the initial population
    # uniformly over the whole search space
    if length(upper) == length(lower)
        ww = upper - lower
        limit_search_space= true
    end
    for i in 1: method.n_particles
        tmp= copy(problem.x_initial)
        random_x!(tmp, n, upper=upper, lower= lower)
        X[:,i] = tmp
        X_best[:,i] = X[:,i]
        if limit_search_space
            # initialise randomly positif or negative vitess that not execced 0.5 from the search space 
            V[:,i] = map(x->x * (rand() * 2 - 1) / 10, ww)
        else #unbounded search space
            V[:,i] = rand(T, n)
        end
    end
    # crush the first inputs to be like initial x
    X[:,1] = copy(problem.x_initial)
    X_best[:,1] = copy(problem.x_initial)
    
    for i in 2:n_particles
        score[i] = problem.objective(X[:, i])
    end

    ParticleSwarmState(
        x,
        score[1],
        0,
        c1,
        c2,
        w,
        limit_search_space,
        X,
        V,
        X_best,
        score,
        best_score,
        x_learn,
        current_state,
        1000)
end


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
                dd += ddd^2
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
                       score)
    
    for i in 1:n_particles
        score[i] = f(X[:, i])
    end
    
end

function update_state!(method::ParticleSwarm, problem::Problem{T}, iteration::Int, state::ParticleSwarmState) where {T}
    nbrSim = 0   
    n = problem.dimension

    state.f_x = housekeeping!(state.score,
                            state.best_score,
                            state.X,
                            state.X_best,
                            state.x,
                            state.f_x,
                            method.n_particles)
    # Elitist Learning:
    # find a new solution named 'x_learn' which is the current best
    # solution with one randomly picked variable being modified.
    # Replace the current worst solution in X with x_learn
    # if x_learn presents the new best solution.
    # In all other cases discard x_learn.
    # This helps jumping out of local minima.
    i_worst = argmax(state.score)
    
    state.x_learn=state.x
    random_index = rand(1:n)
    sigma_learn = 1 - (1 - 0.1) * state.iteration / state.iterations
    r3 = randn() * sigma_learn

    if state.limit_search_space
        state.x_learn[random_index] = round(state.x_learn[random_index] + (problem.upper[random_index] - problem.lower[random_index]) / 3.0 * r3)
        if state.x_learn[random_index] < problem.lower[random_index]
            state.x_learn[random_index] = problem.lower[random_index]
        elseif state.x_learn[random_index] > problem.upper[random_index]
            state.x_learn[random_index] = problem.upper[random_index]
        end
    else
        state.x_learn[random_index] = round(state.x_learn[random_index] + state.x_learn[random_index] * r3)
    end
    
    score_learn = problem.objective(state.x_learn)
    nbrSim +=1

    if score_learn < state.f_x
        state.f_x = score_learn 
        state.X_best[:, i_worst] = state.x_learn
        state.X[:, i_worst] = state.x_learn
        state.x[:] = state.x_learn
        state.score[i_worst] = score_learn
        state.best_score[i_worst] = score_learn
    end

    state.current_state, _f = get_swarm_state(state.X, state.score, state.x, state.current_state)
    #state.w, state.c1, state.c2 = update_swarm_params!(state.c1, state.c2, state.w, state.current_state, _f)
    update_swarm!(state.X, state.X_best, state.x, n, method.n_particles, state.V, state.w, state.c1, state.c2)

    if state.limit_search_space
        limit_X!(state.X, problem.lower, problem.upper, method.n_particles, n)
    end
    compute_cost!(problem.objective, method.n_particles, state.X, state.score)
    nbrSim += method.n_particles
    state.iteration += 1

    collect(state.x), state.f_x, nbrSim
end

function create_state_for_HH(method::ParticleSwarm, problem::Problem, archive)
    nbrSim = 0
    n = length(problem.x_initial)
    upper = problem.upper
    lower = problem.lower
    if 0 < method.n_particles < 3 
        @warn("Number of particles is set to 3 (minimum required)")
        method.n_particles = 3
    elseif method.n_particles==0
        # user did not define number of particles
        n_particles = maximum([3, length(problem.x_initial)])
    end

    X_tmp, score, nbrSim = get_solution_from_archive(archive, problem, n_particles)
    X= X_tmp[1]
    for i in 2: length(X_tmp)
        X= hcat(X, X_tmp[i])
    end

    V = Array{Float64,2}(undef, n, n_particles)
    X_best = copy(X)
    best_score = copy(score)
    
    x = copy(archive.x[argmin(archive.fit)])
    x_learn = copy(x)

    current_state = 0
    limit_search_space= false
    
    # if search space is limited, spread the initial population
    # uniformly over the whole search space
    if length(upper) == length(lower)
        ww = upper - lower
        limit_search_space= true
    end
    for i in 1: method.n_particles
        if limit_search_space
            # initialise randomly positif or negative vitess that not execced 0.5 from the search space 
            V[:,i] = map(x->x * (rand() * 2 - 1) / 10, ww)
        else #unbounded search space
            V[:,i] = rand(T, n)
        end
    end

    ParticleSwarmState(
        x,
        score[1],
        0,
        c1,
        c2,
        w,
        limit_search_space,
        X,
        V,
        X_best,
        score,
        best_score,
        x_learn,
        current_state,
        1000), nbrSim
end
