include("JRP.jl")
include("../../src/SmartOptimizer.jl")
using Main.SmartOptimizer



	# define grid settings
	Lx = 1400.
	Ly = 1400.

	Lsquare = 2.0

	rmin = 40.
	rmax = 140.

	#make grid (new one at each time for illustrations)
	g = Grid(Lx, Ly, Lsquare)

	# make sonar (depends on grid)
	s = Sonar(rmin, rmax, g)


	lb = 0.
	ub = s.rmax*2  

	nvars = 20

	max_length = (nvars+1)*Ly/Lsquare
	maxgen = 100

	
	
	lower = lb .* ones(nvars) #this is because we append 0. to lower: lower = [0. lower]
	upper = ub .* ones(nvars) #this is because we append Ly to upper: upper = [Ly upper]   
	
	lower[1] = 0
	upper[1] = s.rmax 

	
	fobj_horizontal = x -> fobj_horizontal_vector(Float64.(x), g, s, max_length)

	

	p = Problem(fobj_horizontal, false, 20, upper = upper, lower = lower, initial_x = ones(nvars))
	sum(p.x_initial)
	m = ParticleSwarm( n_particles = 100)
	optimize(m, p, Options(max_iterations = 30))
	a = [5,6,8,9,8]
	
   #= m = GeneticAlgorithm(populationSize = 100, crossoverRate =0.8, mutationRate =0.1, epsilon =0.2,
        selection = roulette,
        crossover = line(), 
		mutation = Main.SmartOptimizer.mutation_domainrange) 

		 
	optimize(m, p, Options(max_iterations = 5))
	fobj_horizontal([19,131,58,114,66])
=#
#=	using Evolutionary

	lx = Int64.(lower)
	ux = Int64.(upper)

	cons(x) = sum(x)
	lc = [0]
	uc = [1400]

	tc = [Integer, Integer, Integer, Integer, Integer]

	cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
	c = MixedTypePenaltyConstraints(WorstFitnessConstraints(cb, cons), tc)
	
	opts = Evolutionary.Options(iterations=100, abstol=1e-5)
	init = ()->rand(0:140, 5)
	
	init()

	mthd = GA(populationSize=100, crossoverRate=0.8, mutationRate=0.05, selection=susinv, 
		crossover=MILX_updated(0.0,0.5,0.3), mutation=MIPM(lx,ux))
	result = Evolutionary.optimize(fobj_horizontal, c, init, mthd, opts)
	result.minimizer
	function MILX_updated(μ::Real = 0.0, b_real::Real = 0.5, b_int::Real = 0.3) # location μ, scale b > 0
		function milxxvr(v1::T, v2::T) where {T <: Vector}
			@assert all([typeof(a) == typeof(b) for (a, b) in zip(v1, v2)]) "Types of variables in vectors do not match"
			l = length(v1)
			U, R = rand(l), rand(l)
			B = (isa(x, Integer) ? b_int : b_real for x in v1)
			βs = broadcast((u, r, b) -> r > 0.5 ? μ + b * log.(u) : μ - b * log.(u), U, R, B)
			S = βs .* abs.(v1 - v2)
			c1 = round.(v1 + S)
			c2 = round.(v2 + S)
			return c1, c2
		end
		return milxxvr
	end



v1 = Int64.(ones(5))
v2 = v1.*12	
l = length(v1)
b_real = 0.5
b_int = 0.3
μ = 0.0
U, R = rand(l), rand(l)
B = (isa(x, Integer) ? b_int : b_real for x in v1)
βs = broadcast((u, r, b) -> r > 0.5 ? μ + b * log.(u) : μ - b * log.(u), U, R, B)
S = βs .* abs.(v1 - v2)
c1 = round.(v1 + S)
c2 = round.(v2 + S)
=#
sum([42,66,97,101,43,71,37,46,4,52,33,8,50,103,50,93,42,11,4,91])
fobj_horizontal([0,0,140,140,0,25,140,140,72,48,140,111,36,58,116,96,38,24,140,85])