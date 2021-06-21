include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using DataFrames
using CSV
p = Problem(sim_GG1K,false,10, upper=Integer.(ones(10).*20), lower= Integer.(ones(10)))
x=p.x_initial
methods= DataFrame( abv=[], method=[], duration=[], x=[], fit=[], nbrSim=[] )
push!(methods, ["SC", StochasticComparison(), 0, x, Inf, 0])

"""push!(methods, ["GA", GA(), 0, x, Inf, 0])
push!(methods, ["HJ", HookeAndJeeves(), 0, x, Inf, 0])
push!(methods, ["NM", NelderMead(), 0, x, Inf, 0])
push!(methods, ["PS", ParticleSwarm(), 0, x, Inf, 0])
push!(methods, ["SA", SimulatedAnnealing(), 0, x, Inf, 0])
push!(methods, ["SC", StochasticComparison(), 0, x, Inf, 0])
push!(methods, ["SR", StochasticRuler(), 0, x, Inf, 0])
push!(methods, ["TS", TabuSearch(), 0, x, Inf, 0])
push!(methods, ["GS", GeneratingSetSearcher(), 0, x, Inf, 0])
push!(methods, ["COMPASS", COMPASS_Searcher(), 0, x, Inf])"""

t= time()
#=for i in 1:nrow(methods)
    method= methods[i, :method]
    res=0
    try
        res = optimize(method,p)
    catch e
        println("the ", method.method_name, " is interrupted...")
        msg = sprint(showerror, e)
        println(msg)
        continue
    end
    if res != 0
        methods[i, :duration]=res[1].elapsed_time
        methods[i, :x]=res[1].minimizer
        methods[i, :fit]=res[1].minimum
        methods[i, :nbrSim]=res[2]
    end
end """=#
for i in 1:nrow(methods)
    method= methods[i, :method]
    res=0
    res = optimize(method,p)
    if res != 0
        methods[i, :duration]=res[1].elapsed_time
        methods[i, :x]=res[1].minimizer
        methods[i, :fit]=res[1].minimum
        methods[i, :nbrSim]=res[2]
    end
end 

select!(methods, Not(:method))

@show total_time= time()-t
#CSV.write(methods, "C:\\Users\\Folio\\Desktop\\Preparation doctorat ERM\\Experimental Results\\discrete low level heuristics comparison\\COMPASS1000Iter.csv")
for i in 1:2000 
    println(i)
    sim_GG1K([11, 20, 1, 15, 4, 15, 11, 7, 14, 16])
end