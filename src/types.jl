abstract type HyperHeuristic end 
abstract type HH_State end 

abstract type LowLevelHeuristic end
abstract type State end

struct Options
  ϵ_f::Float64
  ϵ_x::Float64
  max_iterations::Int
  store_trace::Bool
end

function Options(;
  ϵ_f = 1e-16,
  ϵ_x = 1e-16,
  max_iterations = 1000,
  store_trace = false)
  return Options(
    ϵ_f,
    ϵ_x,
    max_iterations,
    store_trace
  )
end

struct Problem{T}
  objective::Function
  x_initial::Array{T}
  continous::Bool
  dimension::Int
  upper::Array{T}
  lower::Array{T}
end

# TODO i must create anothor file called utilities and move this function to it
function random_x!(x::Array{T,1}, dim::Int;  upper=Array{T,1}[], lower=Array{T,1}[]) where {T<:Number}
 
    if length(upper) == 0 && length(lower) == 0
      x =rand(T,dim)
    elseif length(upper) == 0
      x =lower .+  abs.(rand(T,dim))
    elseif length(lower) == 0
      x =upper .-  abs.(rand(T,dim))
    else
      for i in 1:dim
        x[i] = rand(lower[i]:upper[i])
      end
    end
  
end

function Problem(objective::Function, continous::Bool, dimension::Int=1; upper=[], lower=[], initial_x=[], replicationsNbr = 1)
  T= continous ? Float64 : Int64
  if length(initial_x) == 0
    initial_x= Array{T,1}(undef,dimension)
    random_x!(initial_x,dimension, upper=upper, lower=lower)  
  end
  # standarizat the types of upper init_x and lower
  initial_x=convert(Array{T,1}, initial_x)
  upper=convert(Array{T,1}, upper)
  lower = convert(Array{T,1}, lower)
  function obj(x)
      #averaging the objective function
      s = 0
      for i in 1:replicationsNbr
        s += objective(x)
      end
      return s/replicationsNbr
  end
  Problem(obj, initial_x, continous, dimension, upper, lower)
end


struct Results{T} 
  method_name::String
  x_initial::Array{T}
  minimizer::Array{T}
  minimum
  iterations::Int
  converged::Bool
  convergence_criteria::Float64
  elapsed_time::Real
end


struct PerformanceFactors
  ΔFitness::Real # difference between the last and the new solutions
  numSimRun::Integer
  CPUTime::Float64
  avgFitness::Real # if we run the LLH for many iteration, this value will be the average of all the traversed solution
  lastFitness::Real
  #here you can add another factors
end
function Base.show(io::IO, results::Results)
  println(io, "Optimization Results")
  println(io, " * Algorithm: ", results.method_name)
  println(io, " * Minimizer: [",join(results.minimizer, ","),"]")
  println(io, " * Minimum: ", results.minimum)
  println(io, " * Iterations: ", results.iterations)
  println(io, " * Converged: ", results.converged ? "true" : "false")
  println(io, " * Elapsed time: ",results.elapsed_time," seconds") 
  
  return
end
