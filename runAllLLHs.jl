include("src/SmartOptimizer.jl")
include("simulations/GG1K_simulation.jl")
using Main.SmartOptimizer
using DataFrames
p = Problem(sim_GG1K,false,10, upper=Integer.(ones(10).*20), lower= Integer.(ones(10)))
x=p.x_initial
methods= DataFrame( method=[], duration=[], x=[], fit=[], nbrSim=[] )

push!(methods, [GA(), 0, x, Inf, 0])
push!(methods, [HookeAndJeeves(), 0, x, Inf, 0])
push!(methods, [NelderMead(), 0, x, Inf, 0])
#push!(methods, [OCBA(), 0, x, Inf])
push!(methods, [ParticleSwarm(), 0, x, Inf, 0])
push!(methods, [SimulatedAnnealing(), 0, x, Inf, 0])
push!(methods, [StochasticComparison(), 0, x, Inf, 0])
push!(methods, [StochasticRuler(), 0, x, Inf, 0])
push!(methods, [TabuSearch(), 0, x, Inf, 0])
push!(methods, [GeneratingSetSearcher(), 0, x, Inf, 0])
#push!(methods, [COMPASS_Searcher(), 0, x, Inf])
abv = ["GA", "HJ", "NM", "PS", "SA", "SC", "SR", "TS", "TS", "GSS"]
t= time()
for i in 1:nrow(methods)
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
        methods[i, :fit]=res[2]
    end
end 

for i in 1:nrow(methods)
    methods.method[i]= abv[i]
end
@show total_time= time()-t
