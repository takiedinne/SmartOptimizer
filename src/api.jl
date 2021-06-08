function HH_optimize(method::HyperHeuristic, problem::Problem{T}, options::Options) where {T<:Number}
  iteration = 1
  converged = false
  x_cur, x_prev = copy(problem.x_initial), zeros(T, length(problem.x_initial))

  f_cur, f_prev = problem.objective(problem.x_initial), Inf

  # Start timing now
  t1 = time()
  lowLevelHeuristics= loadAllLLH()
  state = initial_HHstate(method, problem)

  while true
    LLHs = selectLLH(method, lowLevelHeuristics, problem, iteration, state)
   
    performances = applyLLHs!(method, LLHs, problem, iteration, state)
    
    learningFunction(method, LLHs, problem, iteration, state, performances)
    if ( iteration >= options.max_iterations)
      break
    end
    iteration += 1
  end

  elapsed_time = time() - t1

  return Results(
    method.method_name,
    problem.x_initial,
    x_cur,
    f_cur,
    iteration,
    converged,
    options.ϵ_x,
    elapsed_time,
    trace
  )
end

function HH_optimize(method::LowLevelHeuristic, problem::Problem{T}) where {T<:Number}
  return optimize(method, problem, Options())
end

function loadAllLLH()
  methods= []
  push!(methods, [GA(), 0, x, Inf])
  push!(methods, [HookeAndJeeves(), 0, x, Inf])
  push!(methods, [NelderMead(), 0, x, Inf])
  #push!(methods, [OCBA(), 0, x, Inf])
  push!(methods, [ParticleSwarm(), 0, x, Inf])
  push!(methods, [SimulatedAnnealing(), 0, x, Inf])
  push!(methods, [StochasticComparison(), 0, x, Inf])
  push!(methods, [StochasticRuler(), 0, x, Inf])
  push!(methods, [TabuSearch(), 0, x, Inf])
  push!(methods, [GeneratingSetSearcher(), 0, x, Inf])
  push!(methods, [COMPASS_Searcher(), 0, x, Inf])
  return methods
end

function apply_LLH!(LLHs, problem::Problem, phaseSize::Integer, HHState::HH_State)
  performances=[]
  newSolutions=[]
  for LLH in LLHs
    state=create_state_for_HH()# we'll see how this  function works after
    prev_fit=HHState.x_fit
    current_fit = HHState.x_fit
    current_x =HHState.x
    nbrSim=0
    #start timing for LLH monitoring
    CPUTime =time()
    for i in 1:phaseSize
      current_x, current_fit, lastnbrSim = update_state!(LLH, problem, iteration, state)
      nbrSim += lastnbrSim
    end
    CPUTime=time()-CPUTime
    # create the performance struct
    Δfitness= current_fit - prev_fit
    performance = PerformanceFactors(Δfitness, nbrSim, CPUTime)
    push!(performances, performance)
    push!(newSolutions, [current_x, current_fit])
  end
  newSolutions, performances
end

function get_solution_from_archive(archive, problem::Problem, nbr_of_solutions::Integer)
  nbrSim=0
  archiveCopy= copy(archive)
  archiveCopy= sort!(archiveCopy,[:fit])
  if nrow(archive) >= nbr_of_solutions
      pop = archiveCopy.x[1:nbr_of_solutions]
      f_pop = archiveCopy.fit[1:nbr_of_solutions]
  else
      pop = archiveCopy.x
      f_pop = archiveCopy.fit
      n = length(archiveCopy.x[1])
      for i in 1:(nbr_of_solutions - nrow(archive))
          tmp = copy(archiveCopy.x[1])
          random_x!(tmp, n, upper=problem.upper, lower=problem.lower)
          push!(pop, tmp)
          push!(f_pop, problem.objective(tmp))
          nbrSim += 1
      end
  end
  if nbr_of_solutions == 1
    return pop[1], f_pop[1], nbrSim
  else
    return pop, f_pop, nbrSim
  end
end

function optimize(method::LowLevelHeuristic, problem::Problem{T}, options::Options) where {T<:Number}
  iteration = 1
  converged = false
  trace = nothing
  x_cur, x_prev = copy(problem.x_initial), zeros(T, length(problem.x_initial))
  
  f_cur, f_prev = problem.objective(problem.x_initial), Inf

  if options.store_trace
    # Set up automatic tracking of objective function evaluations
    trace = create_trace(method)
    problem = setup_trace(problem, trace)
    trace!(trace, 0, x_cur, f_cur)
  end

  # Start timing now
  t1 = time()

  state = initial_state(method, problem)
  nbrTotalSim = 0
  while true
    println(method.method_name," iteration: ", iteration)
    x_cur, f_cur, nbrSim = update_state!(method, problem, iteration, state)
    nbrTotalSim += nbrSim
    
    if options.store_trace
      trace!(method, trace, iteration, x_cur, f_cur, options, state)
    end

    converged = has_converged(method, (x_prev, x_cur), (f_prev, f_cur), options, state)

    if (converged || iteration >= options.max_iterations)
      break
    end

    copy!(x_prev, x_cur)
    f_prev = f_cur
    iteration += 1
  end

  elapsed_time = time() - t1

  return Results(
    method.method_name,
    problem.x_initial,
    x_cur,
    f_cur,
    iteration,
    converged,
    options.ϵ_x,
    elapsed_time,
    trace
  ), nbrTotalSim
end

function optimize(method::LowLevelHeuristic, problem::Problem{T}) where {T<:Number}
  return optimize(method, problem, Options())
end

function create_trace(method::LowLevelHeuristic)
  SearchTrace()
end

function setup_trace(problem::Problem, trace::SearchTrace)
  objective(x) = begin
    value = problem.objective(x)
    push!(trace.evaluations, (copy(x), value))
    return value
  end

  return Problem(objective, problem.x_initial)
end

function trace!(method::LowLevelHeuristic, trace::SearchTrace, i::Int, x::Array{T}, f::T, options::Options, state::State) where {T<:Number}
  trace!(trace, i, x, f)
end

function trace!(trace::SearchTrace, i::Int, x::Array{T}, f::T) where {T<:Number}
  push!(trace.iterations, (copy(x), f))
end

function has_converged(method::LowLevelHeuristic, x::Tuple{Array{T},Array{T}}, f::Tuple, options::Options, state::State) where {T<:Number}
 # return has_converged(x..., options) || has_converged(f..., options)
 return false 
end

function has_converged(f_cur::T, f_prev::T, options::Options) where {T<:Number}
  f_cur - f_prev < options.ϵ_f && println("f converged")
  return f_cur - f_prev < options.ϵ_f
end

function has_converged(x_cur::Array{T}, x_prev::Array{T}, options::Options) where {T<:Number}
  norm(x_cur - x_prev) < options.ϵ_x && println(" x converged")
  return norm(x_cur - x_prev) < options.ϵ_x
end

