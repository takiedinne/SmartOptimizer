struct MarkovChainHH <:HyperHeuristic
    method_name::String
    episodeSize::Integer
    learner::Learner
    moveAcceptance::Function
    archiveSize::Integer
end
MarkovChainHH(;es=1, LF= reward_punish_learner(), MA=NaiveAcceptance, AS=10) = MarkovChainHH("Marcov chain hyper heuristic",
                                                                    es, LF, MA, AS);
                                                                    
mutable struct MarkovChainState{T} <: HH_State
    x_best::Array{T,1} #reference
    x::Array{T,1} #current solution
    x_best_fit
    x_fit

    transitionMatrix::Matrix{Float64}

    # golobal fields because  i didin't find a way to do the inheritance between the types
    LLHs::Array{LowLevelHeuristic,1} #if we want to use Tabu search so not allways the same LLHs list 
    archive::DataFrame

    currentLLHIndex::Integer #index of the LLH it is useful for SARSA reinforcement learning
    previousLLHIndex::Integer
    nextLLHIndex::Integer # for SARSA learning
end
function initial_HHstate(method::MarkovChainHH, problem::Problem)
    x= problem.x_initial
    f= problem.objective(x)
    
    LLHs= loadAllLLH()
    n = length(LLHs)
    transitionMatrix = ones(Float64, n ,n)*1/n #equal
    archive= DataFrame(x=[], fit=Array{Float64,1}())
    push!(archive,[x,f])
    previousLLHIndex= rand(1:n) #choose randomly one LLH to be applie at the first stage
    nextLLHIndex = roulette(transitionMatrix[previousLLHIndex,:],1)[1] #choose the next one 
    HHState = MarkovChainState( x,
                copy(x),
                f,
                f,
                transitionMatrix,
                LLHs,
                archive,
                nextLLHIndex,#current
                previousLLHIndex,#previous
                nextLLHIndex #we need this for SARSA if i would Apply it
                          )
end

function update_HHState!(method::MarkovChainHH, problem::Problem, HHState::MarkovChainState, iteration)
    currentLLH= HHState.LLHs[HHState.currentLLHIndex] 
    #apply the selected LLH 
    #apply_LLH! return array of tuple (new solution_i for LLH_i, fit_i) and array of performance  
    newSolution, performance = apply_LLH!([currentLLH], problem, method.episodeSize, HHState)
    newSolution, performance = newSolution[1], performance[1] # cause we've aoolied only one LLH
    #new solution is tuple of solution and fitness
    #move acceptance
    if method.moveAcceptance(newSolution, [HHState.x, HHState.x_fit])
        HHState.x, HHState.x_fit = newSolution
        #check if it is a new best solution
        if HHState.x_best_fit > newSolution[2]
            HHState.x_best, HHState.x_best_fit = newSolution
        end
    end
    
    #learning mechanism
    learn!(method.learner, performance.ΔFitness, HHState.currentLLHIndex,
            view(HHState.transitionMatrix, HHState.previousLLHIndex, :))
    show(stdout, "text/plain", HHState.transitionMatrix)
    println()
    #select the next LLH to apply
    HHState.previousLLHIndex = HHState.currentLLHIndex
    HHState.currentLLHIndex = roulette(HHState.transitionMatrix[HHState.currentLLHIndex, :],1)[1]

    #update the archive
    update_archive!(method, HHState, newSolution)
    HHState.x , HHState.x_fit
end

function fix_reward_punish(HHstate::MarkovChainState, performances::PerformanceFactors, LLHs)
    #this function give reward or punishment to the applied LLH
    reward = 1
    punish = 0
    if performances.ΔFitness < 0
        HHstate.transitionMatrix[HHstate.previousLLHIndex, HHstate.currentLLHIndex] += reward
    else
        HHstate.transitionMatrix[HHstate.previousLLHIndex, HHstate.currentLLHIndex] -= punish
    end
end
 