#=
about the step size initially i will fixed to 
 0.25 from all the search space  and at each 
 failed we decrease this step size to the half 
=#
struct HookeAndJeeves <: LowLevelHeuristic
    method_name::String
    initial_step_size::Real # percent from overall  the search space by default 25%
    step_reduction::Real # fraction of reduction by default 0.5
    ϵ_h::Real # here the small allowed step size by default 1
end
HookeAndJeeves(;
    initial_step_size = 0.25,
    step_reduction = 0.5,
    ϵ_h = 1.0) = HookeAndJeeves("Hooke and Jeeves", initial_step_size, step_reduction, ϵ_h)

mutable struct HookeAndJeevesState{T} <: State
    current_dim::Int
    step_size::Real
    f_x  #current scor
    x::Array{T,1}#current solution
    x_b::Array{T,1}
end

function initial_state(method::HookeAndJeeves, problem::Problem{T}) where {T<:Number}
   
    return HookeAndJeevesState(
                1,
                method.initial_step_size,
                problem.objective(problem.x_initial),
                copy(problem.x_initial),
                copy(problem.x_initial)
    )
end
  
function has_converged(method::HookeAndJeeves, state::HookeAndJeevesState) 
    # Convergence is based on step size
    return state.step_size < method.ϵ_h
end

function update_state!(method::HookeAndJeeves, problem::Problem{T}, iteration::Int, state::HookeAndJeevesState) where {T}
    # to review this assertion it can causes an error
    @assert (problem.upper != [] && problem.lower != []) || state.step_size >= 1 " the problem must be bounded to use fractional step size..."
    
    n = problem.dimension
    upper = problem.upper
    lower = problem.lower
    x_k, x_b = state.x, state.x_b
    nbrSim = 0
    # Evaluate a positive and a negative point in each cardinal direction
    # and update as soon as one is found
    while state.current_dim <= n
        # Arbitrarily choose the positive direction first
        improved=false
        for dir in [1,-1]
            #calculate step size
            if state.step_size >= 1
                length_step = state.step_size
            else
                length_step = state.step_size * (upper[state.current_dim]-lower[state.current_dim])
            end
            x_trial= copy( x_k)
            x_trial[state.current_dim] += round(dir * length_step)

            #check if the new point in inbounds
            if x_trial[state.current_dim] > upper[state.current_dim]
                x_trial[state.current_dim] = upper[state.current_dim]
            elseif x_trial[state.current_dim] < lower[state.current_dim]
                x_trial[state.current_dim] = lower[state.current_dim]
            end

            f_trial = problem.objective(x_trial)
            nbrSim += 1
    
            # If the point is better, immediately go there
            if (f_trial <= state.f_x)
                copy!(x_k, x_trial)
                state.f_x = f_trial
                state.current_dim += 1
                improved=true
                break
            end
        end
        improved && break
        state.current_dim += 1
    end
    
    # If the cardinal direction searches did not improve, reduce the
    # step size
    if (x_k == x_b)
        #@show "didin't improve at $iteration"
        state.step_size *= method.step_reduction
    end

    # Attempt to move in an acceleration based direction
    x_trial = 2*x_k - x_b
    check_in_bounds(upper, lower, x_trial)
    f_trial = problem.objective(x_trial)
    nbrSim += 1
    copy!(x_b, x_k)
    state.current_dim = 1

    # If the point is an improvement use it
    if f_trial <= state.f_x
        copy!(x_k, x_trial)
        state.f_x = f_trial
    end
    state.x = x_k
    state.x, state.f_x, nbrSim
end

function create_state_for_HH(method::HookeAndJeeves, problem::Problem, HHState::HH_State)
    return HookeAndJeevesState(
                1,
                method.initial_step_size,
                HHState.x_fit,
                copy(HHState.x),
                copy(HHState.x)
                ), 0
end


