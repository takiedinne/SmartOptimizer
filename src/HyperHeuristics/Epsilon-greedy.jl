"""
the learning mechansim is implmented as [1]

[1] S. S. Choong, L. P. Wong, and C. P. Lim, “Automatic design of hyper-heuristic based
 on reinforcement learning,” Inf. Sci. (Ny)., vol. 436–437, pp. 89–107, 2018,
  doi: 10.1016/j.ins.2018.01.005.
"""
function AdaptEpsilon()
end

mutable struct ϵGreedy <: HyperHeuristic
    method_name::String
    ϵ::Real
    episodeSize::Integer # must be in all the HH
    selectionFunction::Function
    adaptEpsilonFunc::Function # in the first step i will not use this functionalty
    moveAcceptance::MoveAcceptanceMechanism
    archiveSize::Integer
    learningMechanism# responsible of monitoring the performance of LLHs
end

ϵGreedy(;ϵ=0.5, episodeSize=1, SF=Epsilon_greedy_selection_mechanism, aE=AdaptEpsilon, MA=NaiveAcceptance(), AS=10, 
        LM= reward_punish_LM()) = ϵGreedy("ϵ-Greedy",ϵ, episodeSize, SF, aE, MA, AS, LM)

mutable struct ϵGreedyState{T} <: HH_State
    x_best::Array{T,1} #reference
    x::Array{T,1} #current solution
    x_best_fit
    x_fit
    # golobal fields because  i didin't find a way to do the inheritance between the types
    LLHs::Array{LowLevelHeuristic,1} #if we want to use Tabu search so not allways the same LLHs list 
    archive::DataFrame
    currentLLHIndex::Integer #index of the LLH it is useful for SARSA reinforcement learning
    learningState    
end

function Epsilon_greedy_selection_mechanism(method::ϵGreedy, HHState::ϵGreedyState)
    random_number = rand()
    currentLLHIndex = 0 
    if random_number < method.ϵ
        #random strategy
        currentLLHIndex=rand(1:length(HHState.LLHs))
    else
        #greedy strategy
        currentLLHIndex = argmax(method.learningMechanism.scores)
    end
    currentLLHIndex
end

function initial_HHstate(method::ϵGreedy, problem::Problem{T}) where {T<:Number}
    #initialise global fields
    LLHs= loadAllLLH()
    archive= DataFrame(x=[], fit=Array{Float64,1}())
    x= copy(problem.x_initial)
    fit=problem.objective(x)
    push!(archive,[x,fit])
    # initialiser the learning mechanism
    # according to reference [1] we select 3 state for Q learning and SARSA mechanism
    # the actions are the LLHs
    init_learning_machanism(method, 3, length(LLHs))
    ϵGreedyState( x, copy(x), fit, fit, LLHs, archive,-1, 1), 1
end 

function update_HHState!(method::ϵGreedy, problem::Problem, HHState::ϵGreedyState, iteration)
    currentLearningState= HHState.learningState
    HHState.currentLLHIndex = method.selectionFunction(method, HHState)
    currentLLH= HHState.LLHs[HHState.currentLLHIndex] 
    #apply the selected LLH 
    #apply_LLH! return array of tuple (new solution_i for LLH_i, fit_i) and array of performance  
    
    newSolution, performance = apply_LLH!([currentLLH], problem, method.episodeSize, HHState)
    newSolution, performance = newSolution[1], performance[1] # cause we've aoolied only one LLH
    
    #new solution is tuple of solution and fitness
    #move acceptance
    if isAccepted(method.moveAcceptance, newSolution, [HHState.x, HHState.x_fit])
        HHState.x, HHState.x_fit = newSolution
        #check if it is a new best solution
        if HHState.x_best_fit > newSolution[2]
            HHState.x_best, HHState.x_best_fit = newSolution
        end
    end
    
    HHState.learningState = getCurrentState(performance) #i must use fitness to precise which state we are
    # for learning function we need the HHstate, performance resulting after applying the LLh and the LLHs applied
    learn!(method.learningMechanism, performance.ΔFitness,currentLearningState, HHState.currentLLHIndex, HHState.learningState)
    
    #update the archive
    update_archive!(method, HHState, newSolution)
    HHState.x , HHState.x_fit, performance.numSimRun
end 
function getCurrentState(performance::PerformanceFactors)
    # according to [1] this algorithm use three state.
    # we use fitness based states and we use normalization step to ensure the cross domain applicability
    # if norm(fit) ∈ []
    norm_f = performance.lastFitness / performance.avgFitness
    if 0.0 <= norm_f < 0.67
        state = 1
    elseif 0.67 <= norm_f < 1.33
        state = 2
    else
        state = 3
    end
    state
end
