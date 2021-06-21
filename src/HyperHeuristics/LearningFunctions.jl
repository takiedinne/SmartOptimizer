function fix_reward_punish(HHstate::HH_State, performances::Array{PerformanceFactors,1}, LLHs)
    #this function give reward or punishment to the applied LLH
    reward = 1
    punish = 1
    for i in 1:length(performances)
        index=findall(x->x.method_name == LLHs[i].method_name, HHstate.LLHs)
        
        if performances[i].ΔFitness > 0
            HHstate.scores[index[1]] += reward
        else
            HHstate.scores[index[1]] -= punish
        end
    end
end