function AdaptEpsilon()
end

mutable struct ϵGreedy <: HyperHeuristic
    method_name::String
    ϵ::Real
    episodeSize::Integer # must be in all the HH
    selectionFunction::Function
    learningFunction::Function
    adaptEpsilonFunc::Function
    moveAcceptance::Function
    archiveSize::Integer
end


ϵGreedy(;ϵ=0.5, episodeSize=1, SF=Epsilon_greedy_selection_mechanism ,LF=fix_reward_punish, aE=AdaptEpsilon, MA=AllMoves, AS=10) =
 ϵGreedy("ϵ-Greedy",ϵ, episodeSize, SF, LF, aE, MA, AS)

mutable struct ϵGreedyState{T} <: HH_State
    x_best::Array{T,1} #reference
    x::Array{T,1} #current solution
    x_best_fit
    x_fit
    scores::Array{Float64, 1}
    # golobal fields because  i didin't find a way to do the inheritance between the types
    LLHs::Array{LowLevelHeuristic,1} #if we want to use Tabu search so not allways the same LLHs list 
    archive::DataFrame

    currentLLHIndex::Integer #index of the LLH it is useful for SARSA reinforcement learning
    nextLLHIndex::Integer
end

function Epsilon_greedy_selection_mechanism(method::ϵGreedy, HHState::ϵGreedyState)
    random_number = rand()
    currentLLHIndex = 0 
    if random_number < method.ϵ
        #random strategy
        currentLLHIndex=rand(1:length(HHState.LLHs))
    else
        #greedy strategy
        currentLLHIndex = argmax(HHState.scores)
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
    scores= zeros(length(LLHs)) # initially all the scores are equal
    
    HHState = ϵGreedyState( x, copy(x), fit, fit, scores, LLHs, archive, -1,-1)
    HHState.currentLLHIndex = method.selectionFunction(method, HHState)
    
    HHState
end 

function update_HHState!(method::ϵGreedy, problem::Problem, HHState::ϵGreedyState, iteration)
    
    currentLLH= HHState.LLHs[HHState.currentLLHIndex] 
    #apply the selected LLH 
    #apply_LLH! return array of tuple (new solution_i for LLH_i, fit_i) and array of performance  
    newSolution, performance = apply_LLH!([currentLLH], problem, method.episodeSize, HHState)
    #new solution is tuple of solution and fitness
    #move acceptance
    if method.moveAcceptance(newSolution, [HHState.x, HHState.x_fit])
        HHState.x, HHState.x_fit = newSolution[1]
        #check if it is a new best solution
        if HHState.x_best_fit > newSolution[1][2]
            HHState.x_best, HHState.x_best_fit = newSolution[1]
        end
    end

    #Selection of the next LLH this is is useful if we use SARSA Epsilon_greedy_selection_mechanism
    HHState.nextLLHIndex = method.selectionFunction(method, HHState)
    # for learning function we need the HHstate, performance resulting after applying the LLh and the LLHs applied
    method.learningFunction(HHState, performance, [currentLLH])
    HHState.currentLLHIndex = HHState.nextLLHIndex
    #update the archive
    update_archive!(method, HHState, newSolution[1])
    HHState.x , HHState.x_fit
end 
