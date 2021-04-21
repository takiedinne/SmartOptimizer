using LinearAlgebra
include("../../../simulations/GG1K_simulation.jl")
# Generating Set Search as described in Kolda2003:
#  Kolda, Tamara G., Robert Michael Lewis, and Virginia Torczon. "Optimization
#  by direct search: New perspectives on some classical and modern methods."
#  SIAM review 45.3 (2003): 385-482.
#
"""
`DirectionGenerator` generates the search directions to use at each step of
a GSS search.
"""
abstract type DirectionGenerator end

struct ConstantDirectionGen <: DirectionGenerator
    directions::Matrix{Float64}

    ConstantDirectionGen(directions) = new(directions)
end

directions_for_k(cg::ConstantDirectionGen, k) =
    cg.directions # For a ConstantDirectionGen it is always the same regardless of k...

# We can easily do a compass search with GSS by generating directions
# individually (+ and -) for each coordinate.
compass_search_directions(n) = ConstantDirectionGen([Matrix{Float64}(I, n,n) -Matrix{Float64}(I, n, n)])

const GSSDefaultParameters = Dict(
    :direction_gen => compass_search_directions, #direction generator
    :InitialStepSizeFactor => 0.5,        # Factor times the minimum search space diameter to give the initial StepSize
    :RandomDirectionOrder => true,  # Randomly shuffle the order in which the directions are used for each step
    :StepSizeGamma => 2.0,          # Factor by which step size is multiplied if improved point is found. Should be >= 1.0.
    :StepSizePhi => 0.5,            # Factor by which step size is multiplied if NO improved point is found. Should be < 1.0.
    :StepSizeMax => 1, # factor of a limit on the step size can be set but is typically not => Inf.
    :DeltaTolerance => 1,       # GSS has converged if the StepSize drops below this tolerance level
)
# TODO apdate this to perform with bounds take the small range of bounds and then multiply by the factor
calc_initial_step_size(lower, upper, stepSizeFactor = 0.5) = stepSizeFactor * minimum(upper.-lower)

#=
Generating Set Search as described in Kolda2003:
  Kolda, Tamara G., Robert Michael Lewis, and Virginia Torczon. "Optimization
  by direct search: New perspectives on some classical and modern methods."
  SIAM review 45.3 (2003): 385-482.
"""=#
struct GeneratingSetSearcher
    direction_gen
    step_size_factor::Float64
    random_dir_order::Bool       # shuffle the order of directions?    # initial step size factor
    step_size_gamma::Float64     # step size factor if improved
    step_size_phi::Float64       # step size factor if no improvement
    step_size_max::Float64       # maximal step size
    step_tol::Float64            # step delta tolerance
end
GeneratingSetSearcher(;direction_gen = GSSDefaultParameters[:direction_gen],
                        step_size_factor = GSSDefaultParameters[:InitialStepSizeFactor],
                        random_dir_order =GSSDefaultParameters[:RandomDirectionOrder],
                        step_size_gamma = GSSDefaultParameters[:StepSizeGamma],
                        step_size_phi = GSSDefaultParameters[:StepSizePhi],
                        step_size_max = GSSDefaultParameters[:StepSizeMax],
                        step_tol = GSSDefaultParameters[:DeltaTolerance])= GeneratingSetSearcher(direction_gen,
                        step_size_factor, random_dir_order, step_size_gamma, step_size_phi, step_size_max,
                         step_tol)
mutable struct GeneratingSetSearcherState{T}
    directions
    n::Int
    k::Int
    step_size::Float64           # current step size
    x::AbstractArray{T,1}
    xfitness::Float64
end
function initial_state(method::GeneratingSetSearcher, lower::AbstractArray{T},
    upper::AbstractArray{T}, objfun, initial_x::AbstractArray{T}) where T
    n= length(initial_x)
    directions= method.direction_gen(n)
    step_size = calc_initial_step_size(lower, upper, method.step_size_factor)
    GeneratingSetSearcherState(directions, n , 0, step_size, copy(initial_x), objfun(initial_x))
end

initial_state(GeneratingSetSearcher(),[1,1,1], [5,5,5], sim_GG1K, [1,2,3])
# by default use RandomBound embedder
function GeneratingSetSearcher(problem::OptimizationProblem, parameters::Parameters)
    params = chain(GSSDefaultParameters, parameters)
    GeneratingSetSearcher(ProblemEvaluator(problem),
                        get(params, :DirectionGenerator, compass_search_directions(numdims(problem))),
                        RandomBound(search_space(problem)),
                        params[:RandomDirectionOrder],
                        params[:InitialStepSizeFactor], params[:StepSizeGamma],
                        params[:StepSizePhi], params[:StepSizeMax],
                        params[:DeltaTolerance])
end



has_converged(state::GeneratingSetSearcherState, method::GeneratingSetSearcher) = state.step_size < method.step_tol
state.step_size
method.step_tol
function check_in_bounds(upper, lower, x)
    @assert length(upper)== length(lower) == length(initial_x)
    for i in 1: length(x)
        if x[i]>upper[i]
            x[i]=upper[i]
        elseif x[i]<lower[i]
            x[i]=lower[i]
        end
    end
end

lower=[1,1,1]
upper=[5,5,5]
f=sim_GG1K
function GeneratingSetSearcherAlgo(f, lower, upper)
    @assert length(lower) == length(upper)
    dim = length(lower) 
    # initialisation
    initial_x=Array{Int,1}(undef,dim)
    for i in 1:dim
        initial_x[i]=lower[i]+ round(rand()*(upper[i]-lower[i]))
    end
    method= GeneratingSetSearcher()
    
    state= initial_state(method, lower, upper, f, initial_x)

    println(state.step_size)
    println(method.step_tol)
    println( )
    while !(state.step_size < method.step_tol) && state.k<500

        # Get the directions for this iteration
        @show state.k += 1
        directions = state.directions.directions

        # Set up order vector from which we will take the directions after possibly shuffling it
        order = collect(1:size(directions)[2])
        if method.random_dir_order
            shuffle!(order)
        end

        # Check all directions to find a better point; default is that no one is found.
        found_better = false
        candidate = zeros(state.n, 1)
        f_candidate=Inf
        # Loop over directions until we find an improvement (or there are no more directions to check).
        for direction in order
            candidate = round.(state.x + state.step_size .* directions[:, direction])
            #check if the new point in inbounds
            check_in_bounds(upper, lower, candidate)
            f_candidate = f(candidate)

            if f_candidate<state.xfitness
                found_better = true
                break
            end
        end

        if found_better
            state.x = candidate
            state.xfitness = f_candidate
            state.step_size *= method.step_size_gamma
        else
            state.step_size *= method.step_size_phi
        end
        state.step_size = min(state.step_size, method.step_size_max* minimum(upper.-lower))
        
    end
    return state
end
s=GeneratingSetSearcherAlgo(sim_GG1K, [1,1,1], [5,5,5])
