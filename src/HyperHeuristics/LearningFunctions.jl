abstract type LearningMethod end

mutable struct reward_punish_LM <:LearningMethod
    r #reward
    p #punish
    γ #the threshold
end
reward_punish_LM(;r=1, p=1, gamma=0)= reward_punish_LM(r, p, gamma)
function learn!(learning_mechanism::reward_punish_LM, ΔFitness::Float64, LLHIndex::Integer, scores)
    if ΔFitness < learning_mechanism.γ
        scores[LLHIndex] += learning_mechanism.r
    else
        scores[LLHIndex] -= learning_mechanism.p
    end
end
################################################################
""" 
to model the hyper heuristic as reinforcement learning problem we must precise sveral point namely
the state, the reward signal, the action. 
- the reward signal is 1 if there is an improvement 0 otherwise
- the action is the Set of LLHs (we choose only one on each iteration)
- the state we can categorize the range of the fitness values on n interval see [1] as an example
- for the first version of this method I'll consider n=1 it means we have only one state
in this reinforcement learning method it is imperatively to use current and next LLH in HHState
[1]: Choong, S. S., Wong, L. P., & Lim, C. P. (2018). Automatic design of hyper-heuristic
     based on reinforcement learning. Information Sciences,
     436–437, 89–107. https://doi.org/10.1016/j.ins.2018.01.005
"""
mutable struct SARSA_LM <:LearningMethod
    γ # discount
    α # learning rate
    r # reward signal
    p # punish signal
end
SARSA_LM(;γ= 0.8, α=0.9, r=1, p=0)= SARSA_LM(γ, α, r, p)
function learn!(learning_mechanism::SARSA_LM, ΔFitness::Float64,
                     LLHIndex::Integer, StateActionsScores, nextLLHIndex::Integer, nextStateActionsScores)
    # scores it the actuel state-actions scores 
    if ΔFitness < 0
        reward_signal = learning_mechanism.r
    else
        reward_signal = learning_mechanism.p
    end
    StateActionsScores[LLHIndex] +=  learning_mechanism.α * 
                                    (reward_signal + learning_mechanism.γ * nextStateActionsScores[nextLLHIndex] 
                                    - StateActionsScores[LLHIndex])
    
end
#***********************************************************************
"""
the Q learning mechanism is implemented based on the details from [1] and [2]
and the parameters default values have token from [3]
[1]: Gane, G. P., Horabin, I. S., & Lewis, B. N. (1966). Algorithms for decision making. 
     The Proceedings of the Programmed Learning Conference, 481–502.
[2]: Barto, R. S. S. and A. G. (1967). Reinforcement Learning: An Introduction second edition.
[3]: Choong, S. S., Wong, L., & Lim, C. P. (2018). Automatic Design of Hyper-heuristic based on Reinforcement Learning.
     Information Sciences. https://doi.org/10.1016/j.ins.2018.01.005
"""
mutable struct QLearning_LM <:LearningMethod
    γ # discount
    α # learning rate
    r # reward
    p # punish
end
QLearning_LM(;gamma=0.8, alpha=0.9, r=1, p=0)= QLearning_LM(gamma, alpha, r, p)
function learn!(learning_mechanism::QLearning_LM, ΔFitness::Float64, LLHIndex::Integer,
                    StateActionsScores,  nextStateActionsScores)
    # scores it the actuel state-actions scores 
    if ΔFitness < 0
        reward_signal = learning_mechanism.r
    else
        reward_signal = learning_mechanism.p
    end
    StateActionsScores[LLHIndex] +=  learning_mechanism.α * 
                                    (reward_signal + learning_mechanism.γ * maximum(nextStateActionsScores) 
                                    - StateActionsScores[LLHIndex])
    
end

#this function will be removed after i adapt the epsilon greedy hh with the learning_mechanism
function fix_reward_punish(HHstate::HH_State, performances::Array{PerformanceFactors,1}, LLHs)
    #this function give reward or punishment to the applied LLH
    reward = 1
    punish = 1
    for i in 1:length(performances)
        index=findall(x->x.method_name == LLHs[i].method_name, HHstate.LLHs)
        
        if performances[i].ΔFitness < 0
            HHstate.scores[index[1]] += reward
        else
            HHstate.scores[index[1]] -= punish
        end
    end
end

function SARSA_RL(HHstate::HH_State, performances::Array{PerformanceFactors,1}, LLHs)
    """ 
    to model the hyper heuristic as reinforcement learning problem we must precise several point namely
    the state, the reward signal, the action. 
    - the reward signal is 1 if there is an improvement 0 otherwise
    - the action is the Set of LLHs (we choose only one on each iteration)
    - the state we can categorize the range of the fitness values on n interval see [1] as an example
    - for the first version of this method I'll consider n=1 it means we have only one state
    in this reinforcement learning method it is imperatively to use current and next LLH in HHState
    [1]: Choong, S. S., Wong, L. P., & Lim, C. P. (2018). Automatic design of hyper-heuristic
         based on reinforcement learning. Information Sciences,
         436–437, 89–107. https://doi.org/10.1016/j.ins.2018.01.005
    """
    reward = 1
    punish = 0 
    
    #canstant for the SARSA mechanism their values have taken from [1]
    α = 0.9
    γ = 0.8
    #here we have only one performance  
    if performances[1].ΔFitness < 0
        r = reward
    else
        r = punish
    end

    HHstate.scores[HHstate.currentLLH] = HHstate.scores[HHstate.currentLLH] + α * (r + γ * HHstate.scores[HHstate.nextLLH] - HHstate.scores[HHstate.currentLLH])
    
end

