function HH_optimize(method::HyperHeuristic, problem::Problem{T}, options::Options) where {T<:Number}
  nbrTotalSim = 0
  fit_historic = []
  best_fit_historic = []
  iteration = 1
  converged = false
  trace = nothing
  x_cur, x_prev = copy(problem.x_initial), zeros(T, length(problem.x_initial))

  f_cur, f_prev = problem.objective(problem.x_initial), Inf
  nbrTotalSim += 1
  push!(fit_historic,f_cur)
  push!(best_fit_historic,f_cur)
  # Start timing now
  t1 = time()
  HHstate, nbrOfRuns = initial_HHstate(method, problem)
  nbrTotalSim += nbrOfRuns
  while true
    x_cur, f_cur, nbrOfRuns = update_HHState!(method, problem, HHstate, iteration)
    nbrTotalSim += nbrOfRuns
    push!(fit_historic,HHstate.x_fit)
    push!(best_fit_historic,HHstate.x_best_fit)
    if ( iteration >= options.max_iterations)
      break
    end
    if iteration % 20 == 0 
      #plot the Results
      plot(1:(iteration+1), fit_historic, label="current")
      display(plot!(1:(iteration+1), best_fit_historic, label="best"))
    end
    iteration += 1
  end
  elapsed_time = time() - t1
  
  #=anim = @animate for i ∈ 1:(iteration+1)
      plot(1:i, fit_historic[1:i], label= method.method_name)
  end every 10
  gif(anim, string("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\HyperHeuristic\\",method.method_name,"_",method.moveAcceptance,"_",method.learningMachanism,".gif"), fps = 20)
  
  #plot(1:length(fit_historic), fit_historic, label= string(method.method_name, "-", method.learningMechanism.method_name,"-", method.moveAcceptance))
  plot(1:length(fit_historic), fit_historic, label= string(method.method_name, "-", method.moveAcceptance))
  savefig(string("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\HyperHeuristic\\TabuSearchHH\\",method.method_name,"_",method.moveAcceptance,".png"))
=#
 return Results(
    method.method_name,
    problem.x_initial,
    HHstate.x_best,
    HHstate.x_best_fit,
    iteration,
    converged,
    options.ϵ_x,
    elapsed_time,
    trace
  ), nbrTotalSim
end

function HH_optimize(method::HyperHeuristic, problem::Problem{T}) where {T<:Number}
  return HH_optimize(method, problem, Options())
end

function loadAllLLH()
  methods= Array{LowLevelHeuristic,1}()
  push!(methods, GA())
  push!(methods, HookeAndJeeves())
  push!(methods, NelderMead())
  push!(methods, ParticleSwarm())
  push!(methods, SimulatedAnnealing())
  push!(methods, StochasticComparison())
  push!(methods, StochasticRuler())
  push!(methods, TabuSearch())
  push!(methods, GeneratingSetSearcher())
  push!(methods, COMPASS_Searcher())
  return methods
end

function apply_LLH!(LLHs, problem::Problem{T}, phaseSize::Integer, HHState::HH_State) where T
  performances=Array{PerformanceFactors,1}()
  newSolutions=Array{Tuple{Array{T,1}, Float64},1}()
  for LLH in LLHs
    nbrSim=0
    state, nbrSim = create_state_for_HH(LLH, problem,HHState)# we'll see how this  function works after
    prev_fit= HHState.x_fit
    current_fit = HHState.x_fit
    current_x = HHState.x
    
    #start timing for LLH monitoring
    
    CPUTime = time()
    for i in 1:phaseSize
      current_x, current_fit, lastnbrSim = update_state!(LLH, problem, i, state)
      nbrSim += lastnbrSim
    end
    CPUTime = time()-CPUTime
    println(LLH.method_name, " is applied")
    # create the performance struct
    Δfitness= current_fit - prev_fit
    performance = PerformanceFactors(Δfitness, nbrSim, CPUTime)
    push!(performances, performance)
    push!(newSolutions, (current_x, current_fit))
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
          random_x!(tmp, n, upper = problem.upper, lower = problem.lower)
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

function update_archive!(method::HyperHeuristic, state::HH_State, newSolution)
  # we assume that the archive will maintains n best found solution
  if newSolution[1] in state.archive.x 
    return
  end
  
  if nrow(state.archive) < method.archiveSize
    #here we add directely the new solution
    push!(state.archive, newSolution)
  elseif newSolution[2] < maximum(state.archive.fit)
    #here we remplace te worstest solution
    state.archive[argmax(state.archive.fit),:] = newSolution  
  end
end

function optimize(method::LowLevelHeuristic, problem::Problem{T}, options::Options) where {T<:Number}
  fit_historic=[]
  iteration = 1
  converged = false
  trace = nothing
  x_cur, x_prev = copy(problem.x_initial), zeros(T, length(problem.x_initial))
  
  f_cur, f_prev = problem.objective(problem.x_initial), Inf
  push!(fit_historic,f_cur)
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
    push!(fit_historic,f_cur)
    if options.store_trace
      trace!(method, trace, iteration, x_cur, f_cur, options, state)
    end

    converged = has_converged(method, (x_prev, x_cur), (f_prev, f_cur), options, state)

    if (converged || iteration >= options.max_iterations)
      break
    end

    copy!(x_prev, x_cur)
    f_prev = f_cur
    #=if iteration % 20 == 0 
      #plot the Results
      display(plot(1:(iteration+1), fit_historic))
    end=#
    iteration += 1
  end

  elapsed_time = time() - t1
  #=anim = @animate for i ∈ 1:(iteration+1)
      plot(1:i, fit_historic[1:i], label= method.method_name)
  end every 10
  gif(anim, string("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\",method.method_name,"_currentSolution.gif"), fps = 10)
  
  plot(1:length(fit_historic), fit_historic, label= method.method_name)
  savefig(string("C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\500Iter\\",method.method_name,".png"))
  =#
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

