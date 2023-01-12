using StaticArrays
using PolygonOps
using Polynomials:fit

using JLD2
using Plots

using Logging

using BenchmarkTools
import Base.show
using LinearAlgebra

using Evolutionary, Test, Random



function compute_bottom_factor(Nx::Int64, Ny::Int64; kwargs...)
    
    bottom_factor = fill("sand", (Ny, Nx))

    xv1 = [   0.,    0.,  1400.,  1400.,    0.] * Nx/1400.
    yv1 = [   0., 1400.,  1400.,     0.,    0.] * Ny/1400.
    

    polygon1 = SVector.(xv1,yv1)
    
    xa = 1:Nx
    ya = 1:Ny
    
    points = vec(SVector.(transpose(xa),ya))
    
    inside1 = [inpolygon(p, polygon1; in=true, on=true, out=false) for p in points]
    
    bottom_factor[inside1] .= "sand" 
    
    
    return bottom_factor
    
end

"""
    Grid

Grid structure using a mesh grid that covers the physical space. The area to be covered by the sensor 
is decomposed into small grid cells where the size of the cells is sufficiently small that the
coverage can be treated as uniform over the cell. 

### Fields:
- Lx: width of physical terrain [m]
- Ly: height of physical terrain [m]
- Lsquare: size of a grid cell [m]
- Nx: number of grid cells in x-direction
- Ny: number of grid cells in y-direction
"""
struct Grid
    
    Lx            ::Float64
    Ly            ::Float64
    Lsquare       ::Float64
    Nx            ::Int64
    Ny            ::Int64
    bottom_factor ::Array{String, 2}
    
    """
        Grid(Lx,Ly,lSquare; margin::Float64=0)

    Generate a grid from a physical rectangle of dimensions (Lx, Ly) making use a square 
    mesh with length lSquare.

    ### Notes
    no margin used (contrast with initial implementation)
    """
    function Grid(Lx::Float64, Ly::Float64, Lsquare::Float64; bottomfunction::Function=compute_bottom_factor, kwargs...)
        
        Nx = round(Int, Lx/Lsquare)
        Ny = round(Int, Ly/Lsquare)
        #bottom_factor =  ones(Float64, Ny, Nx)
        bottom_factor = bottomfunction(Nx, Ny; kwargs...)
        
        return new(Lx, Ly, Lsquare, Nx, Ny, bottom_factor)
        
    end
    
end


show(io::IO, g::Grid) = print(io, "A grid of size $(g.Nx)x$(g.Ny) cells")




"""
    Prob

Prob structure computes the probality of detection resulting from the current pass
or the cumulated probality of detections by taking into account previous (overlapping)
passes in addition to the current one.

### Fields:
- covered: cells that are covered by the current pass
- prob_local : probability of detection after current pass
- prob_global: cumulated probability of detection after previous (overlapping) passes
"""
mutable struct Prob

    Nx             ::Int64
    Ny             ::Int64


    covered        ::Array{Bool, 2}
    covered_global ::Array{Bool, 2}
    
    prob           ::Array{Float64, 2}
    prob_global    ::Array{Float64, 2}

    """
        Prob(g)

    Generates arrays storing covered cells and probabilities of detection (for current pass or previous ones).

    ### Notes
    no margin used (contrast with initial implementation)
    """
    function Prob(g::Grid)
        
        # set field equal to zeros rows for vertical, colums for horizontal
        covered        = zeros(Bool   , g.Ny, g.Nx)
        covered_global = zeros(Bool   , g.Ny, g.Nx)
        
        prob           = zeros(Float64, g.Ny, g.Nx)
        prob_global    = zeros(Float64, g.Ny, g.Nx)

        return new(g.Nx, g.Ny,covered, covered_global, prob, prob_global)
        
    end
    
end
    

show(io::IO, p::Prob) = print(io, "A proba of size $(p.Nx) x $(p.Ny) cells")


"""
    Sonar

The AUV collects sonar data from rmin to rmax away from the AUV in both 
the port and starboard directions. We assume sonar swath coverage is always 
achieved from rmin to rmax.The coverage performance is quantified by the 
lateral range curve. 

The lateral range curve defines the probability that a target at a specified 
lateral range from a sonar’s track will be detected along that track and 
subsequently correctly classified as being suggestive of a mine.

### Fields:
- rmin: minimal detection range [grid cells]
- rmax: maximal detection range [grid cells]
- lrc: lateral range curve profile
"""
struct Sonar
    
    rmin        ::Float64
    rmax        ::Float64

    """
        Sonar(rmin::Float64, rmax::Float64, lrc::Array{Float64,1})

    From the actual sonar characteristics [m] get the sonar range in [grid cells@]
    """
    function Sonar(rmin::Float64, rmax::Float64, g::Grid) 
        return new(rmin/g.Lsquare, rmax/g.Lsquare)
    end
    
end

show(io::IO, s::Sonar) = print(io, "A sonar with range $(s.rmin)-$(s.rmax)+1 cells")




#=
"""
    lrc_sand(range::Float64, s::Sonar)

Lateral range curve for sand bottom.
"""

function lrc_sand(range::Float64, s::Sonar)
    
    step = (s.rmax - s.rmin)/5.0
    
    if s.rmin <= range < s.rmin + step
        pd = 0.80
    elseif s.rmin +   step <= range <  s.rmin + 2*step
        pd = 1.00
    elseif s.rmin + 2*step <= range <  s.rmin + 3*step
        pd = 0.90
    elseif s.rmin + 3*step <= range <  s.rmin + 4*step
        pd = 0.85
    elseif s.rmin + 4*step <= range <= s.rmax
        pd = 0.80
    else 
        pd = 0.0
    end
    
end



"""
    lrc_ripples(range::Float64, s::Sonar)

Lateral range curve for ripples bottom.
"""

function lrc_ripples(range::Float64, s::Sonar)
    
    step = (s.rmax - s.rmin)/5.0
    
    if s.rmin <= range < s.rmin + step
        pd = 0.70
    elseif s.rmin +   step <= range <  s.rmin + 2*step
        pd = 0.55
    elseif s.rmin + 2*step <= range <  s.rmin + 3*step
        pd = 0.70
    elseif s.rmin + 3*step <= range <  s.rmin + 4*step
        pd = 0.40
    elseif s.rmin + 4*step <= range <= s.rmax
        pd = 0.20
    else 
        pd = 0.0
    end
    
end



"""
    lrc_mud(range::Float64, s::Sonar)

Lateral range curve for mud bottom.
"""

function lrc_mud(range::Float64, s::Sonar)
    
    step = (s.rmax - s.rmin)/5.0
    
    if s.rmin <= range < s.rmin + step
        pd = 0.75
    elseif s.rmin +   step <= range <  s.rmin + 2*step
        pd = 0.90
    elseif s.rmin + 2*step <= range <  s.rmin + 3*step
        pd = 0.80
    elseif s.rmin + 3*step <= range <  s.rmin + 4*step
        pd = 0.90
    elseif s.rmin + 4*step <= range <= s.rmax
        pd = 0.70
    else 
        pd = 0.0
    end
    
end



"""
    lrc_default(range::Float64, s::Sonar)

Lateral range curve for default bottom.
"""

function lrc_default(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 1.0
    else 
        pd = 0.0
    end
    
end

=#


function lrc_sand(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 0.87
    else 
        pd = 0.0
    end
    
end

function lrc_ripples(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 0.51
    else 
        pd = 0.0
    end
    
end

function lrc_mud(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 0.81
    else 
        pd = 0.0
    end
    
end

function lrc_default(range::Float64, s::Sonar)
    
    if s.rmin <= range <= s.rmax
        pd = 1.0
    else 
        pd = 0.0
    end
    
end





"""
    singletrack_horizontal!(d::Int64, g::Grid, s::Sonar, pdc::Prob)

Compute the detection probability for a single track with sonar s on grid g and proba pdc.
"""

function singletrack_horizontal!(d::Float64, g::Grid, s::Sonar, pdc::Prob)
    
    @debug "single track"

    pdc.covered        = zeros(Bool   , g.Ny, g.Nx)
    pdc.prob           = zeros(Float64, g.Ny, g.Nx)
    
    #Threads.@threads for x in 1:g.Nx
    for x in 1:g.Nx

        for y in max(1,round(Int, d-s.rmax+1)):min(round(Int, d-s.rmin),g.Ny)
            
            @inbounds pdc.covered[y,x] = true

            if     g.bottom_factor[y,x] == "sand"
                @inbounds pdc.prob[y,x] = lrc_sand(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "ripples"
                @inbounds pdc.prob[y,x] = lrc_ripples(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "mud"
                @inbounds pdc.prob[y,x] = lrc_mud(abs(Float64(y)-d),s)
            else
                @inbounds pdc.prob[y,x] = lrc_default(abs(Float64(y)-d),s)
            end                    

        end
        
        for y in max(1,round(Int, d+s.rmin+1)):min(round(Int, d+s.rmax),g.Ny)
            
            @inbounds pdc.covered[y,x] = true

            if     g.bottom_factor[y,x] == "sand"
                @inbounds pdc.prob[y,x] = lrc_sand(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "ripples"
                @inbounds pdc.prob[y,x] = lrc_ripples(abs(Float64(y)-d),s)
            elseif g.bottom_factor[y,x] == "mud"
                @inbounds pdc.prob[y,x] = lrc_mud(abs(Float64(y)-d),s)
            else
                @inbounds pdc.prob[y,x] = lrc_default(abs(Float64(y)-d),s)
            end                    

        end
        
        
        
        
    end
    
    pdc.covered_global .= max.(pdc.covered_global, pdc.covered)    
    
    # update global prob value
    #pdc.prob_global .= pdc.prob_global .+(1.0 .-pdc.prob_global).*pdc.prob
    pdc.prob_global .= max.(pdc.prob_global, pdc.prob)
end



"""
    multitrack_horizontal!(waypoints::Array{Tuple{Int64, Int64},1}, g::Grid, s::Sonar, pdc::Prob)

Compute the detection probability for a set of horizontal tracks, with sonar s on grid g and proba pdc

See also [`singletrack_horizontal!`](@ref)
"""

function multitrack_horizontal!(d::Array{Float64,1}, g::Grid, s::Sonar, pdc::Prob)
        
    @debug "multiple track "
    for i in 1:length(d)
            # get individual contribution
            singletrack_horizontal!(sum(d[1:i]), g::Grid, s::Sonar, pdc::Prob)
    end

    #minimize -fun = maximize fun
    return -sum(pdc.prob_global)/(pdc.Nx*pdc.Ny)    

end



"""
    horizontal_pattern_vector(;d::Array{Int64,1}, g::Grid)

Generate waypoints in a horizontal lawnmower pattern (horizontal scan)
"""

function horizontal_pattern_vector(;d::Array{Float64,1}, g::Grid)

    waypoints_scan   = Array{Tuple{Float64, Float64},2}(undef,0,2)
    waypoints_travel = Array{Tuple{Float64, Float64},2}(undef,0,2)
    spacing          = Array{Float64}(undef,0)
    
    # complete list
    
    for i = 1:length(d)        
        
        if sum(d[1:i]) <= Float64(g.Ny)
            
                    waypoints_scan       = vcat(waypoints_scan  , [(0.  ,  sum(d[1:i  ]))   (Float64(g.Nx),  sum(d[1:i]))])  
            
            if (mod(i,2) == 1)
                
                if (i == 1 )
                    
                    waypoints_travel     = vcat(waypoints_travel, [(0.  ,  sum(d[1:i  ]))   (Float64(g.Nx),  sum(d[1:i]))])
                
                else
                    
                    if d[i] != 0
                        waypoints_travel = vcat(waypoints_travel, [(0.  ,  sum(d[1:i-1]))   (0.           ,  sum(d[1:i]))])              
                    end
                         
                    waypoints_travel     = vcat(waypoints_travel, [(0.  ,  sum(d[1:i  ]))   (Float64(g.Nx),  sum(d[1:i]))])               
                
                end
            
            else

                    if d[i] != 0
                        waypoints_travel = vcat(waypoints_travel, [(Float64(g.Nx),  sum(d[1:i-1]))     (Float64(g.Nx),  sum(d[1:i]))])              
                    end
                
                    waypoints_travel     = vcat(waypoints_travel, [(Float64(g.Nx),  sum(d[1:i  ]))     (0.           ,  sum(d[1:i]))])               
                                   
            end
        
            spacing = vcat(spacing, d[i])

        end
    end
    
    return (waypoints_scan, waypoints_travel, spacing)

end



function getineqH(g)
    if g <=0
        H=0
    else
        H=1
    end
end



function fobj_horizontal_vector(x::Array{Float64,1}, g::Grid, s::Sonar, max_length::Float64 = 10.0^4)
        
    (waypoints_scan, waypoints_travel, spacing) = horizontal_pattern_vector(d=x, g=g)
    
    pdc = Prob(g)
    #println(" for $x pdc = ", sum(pdc.prob_global))
    # determine the track and coverage
     y = multitrack_horizontal!(x, g, s, pdc) 
    
    # Inequality constraints

    # Penalty constant
    lambda  = 10^15

    # Maximum travelling distance allowed         

     ineq1   = sum([norm(collect(row[1]) - collect(row[2])) for row in eachrow(waypoints_travel)]) - max_length
     ineq2   = sum(x) - g.Ly/g.Lsquare
    #ineq3   = g.Ly/g.Lsquare - s.rmax - sum(x) 


   
    #return y + lambda*ineq1^2*getineqH(ineq1) + lambda*ineq2^2*getineqH(ineq2) 
    return y + lambda*ineq2^2*getineqH(ineq2) 


end



mutable struct gaDatVar

    #    
    # Data structure: Genetic Algorithm parameters
    #


    # Parameters that have to be defined by user
    #
    # FieldD=[lb; ub]        # lower (lb) and upper (up) bounds of the search space. 
    #                        # each dimension of the search space requires bounds 
    # Objfun="costFunction"  # Name of the 0bjective function to be minimize
    #  
    
    
    # Parameters that could be defined by user, in other case, there is a default value
    #
    # MAXGEN={NVAR*20+10}    # Number of generation, NVAR*20+10 by default
    # NIND={NVAR*50}         # Size of the population, NVAR*50 by default
    # alfa=0                 # Parameter for linear crossover, 0 by default
    # Pc=0.9                 # Crossover probability, 0.9 by default
    # Pm=0.1                 # Mutation probability, 0.1 by default
    # ObjfunPar=[]           # Additional parameters of the objective function
    #                        # have to be packed in a structure, empty by default
    # indini=[]              # Initialized members of the initial population, empty
    #                        # by default
    
    FieldD    ::Array{Float64,2}
    Objfun    ::Function

    NVAR      ::Int64
    MAXGEN    ::Int64
    NIND      ::Int64

    alfa      ::Float64
    Pc        ::Float64
    Pm        ::Float64

    indini    ::Array{Float64,2}
    
    Chrom     ::Array{Float64,2}
    ObjV      ::Array{Float64,1}
    xmin      ::Array{Float64,1}
    fxmin     ::Float64
    xmingen   ::Array{Float64,2}
    fxmingen  ::Array{Float64,2}
    rf        ::UnitRange{Int64}
    gen       ::Int64


    function gaDatVar(FieldD,Objfun;
        NVAR       = size(FieldD,2),
        MAXGEN     = NVAR*20+10,
        NIND       = NVAR*50,
        alfa       = 0.,
        Pc         = 0.8,
        Pm         = 0.2,
        indini     = Array{Float64}(undef, 0   ,NVAR),
        Chrom      = Array{Float64}(undef, NIND,NVAR),
        ObjV       = Array{Float64}(undef, NIND),
        xmin       = Array{Float64}(undef, NVAR),
        fxmin      = Inf,
        xmingen    = Array{Float64}(undef, MAXGEN+1,NVAR),
        fxmingen   = Array{Float64}(undef, MAXGEN+1,1   ),
        rf         = 1:NIND,
        gen        = 0)
        
        return new(FieldD,Objfun,NVAR,MAXGEN,NIND,alfa,Pc,Pm,
        indini,Chrom,ObjV,xmin,fxmin,xmingen,fxmingen,rf,gen)

    end


end


function crtrp!(gaDat::gaDatVar)

    # A random real value matrix is created coerced by upper and 
    # lower bounds
    
    aux  = rand(gaDat.NIND,gaDat.NVAR)

    m    = [-1. 1.]*gaDat.FieldD
    
    ublb = ones(gaDat.NIND,1)*m 
    lb   = ones(gaDat.NIND,1)*transpose(gaDat.FieldD[1,:])
    
    gaDat.Chrom = ublb.*aux+lb

end

function gaiteration(gaDat::gaDatVar)

    # Optional user task executed at the end of each iteration
    #
    # For instance, results of the iteration

    println("   ------------------------------------------------")
    println("   Iteration: $(gaDat.gen)")
    println("   xmin: $(gaDat.xmin) -- f(xmin): $(gaDat.fxmin)")
    println("   ------------------------------------------------")
    
end



function garesults(gaDat::gaDatVar)

    # Optional user task executed when the algorithm ends
    #
    # For instance, final result

    println("   ------------------------------------------------")
    println("   ##################   RESULT   ##################")
    println("   Objective function for fxmin: $(gaDat.fxmin)")
    println("   xmin: $(gaDat.xmin)")
    println("   ------------------------------------------------")

end

""""
    ranking(ObjV::Array{Float64,1},RFun::UnitRange{Int64})

idée générale de la fonction et du pourquoi
"""
function ranking(ObjV::Array{Float64,1})

    # Ranking function
    return sortperm(ObjV, rev=false)
end



function ranking(ObjV::Array{Float64,1},RFun::UnitRange{Int64})

    # Ranking function

    if !(length(ObjV)==length(RFun))
        error("RFun have to be of the same size than ObjV.")
    end
    
    val,pos  = sort(vec(ObjV)),sortperm(vec(ObjV))
       
    FitV = zeros(Int,length(ObjV))
    FitV[pos] = reverse((vec(RFun)), dims = 1)


    return FitV

end







function sus2(FitnV::Array{Int64,1}, Nsel::Int64)
    
    # Position of the roulette pointers

    j = 0
    sumfit = 0.
    
    step   = sum(FitnV)/Nsel # distance between pointers
    offset = rand()*step      # offset of the first pointer
    
    NewChrIx = zeros(Int,Nsel)
    
    for i in 1:Nsel
        
        sumfit = sumfit + FitnV[i]
        
        while (sumfit >= offset)
            
            j = j+1
            NewChrIx[j] = i
            offset = offset + step
            
        end
        
    end
    
    return NewChrIx
end


function select(gaDat::gaDatVar,FitnV::Array{Int64,1})
    
    
    indices = sus2(FitnV,round(length(FitnV)))  
    
    SelCh = gaDat.Chrom[indices,:]

    # Disorder the population
    #[kk,indi] = sort(rand(length(FitnV),1))
       
    temp = vec(rand(length(FitnV)))
    
    kk,indi   = sort(temp),sortperm(temp)
    SelCh     = SelCh[indi,:]

    return SelCh
end



function lxov(OldChrom::Array{Float64,2}, XOVR::Float64 = 0.7, alpha::Float64 = 0.)

    # Linear crossover
    # Produce a new population by linear crossover and XOVR crossover probability
    #   NewChroms =lxov(OldChrom, XOVR, alpha, FieldDR)
    #
    # Linear recombination.
    # Parameters "beta1" and "beta2" are randomly obtained inside [-alpha, 1+alpha]
    # interval
    #   Child1 = beta1*Parent1+(1-beta1)*Parent2
    #   Child2 = beta2*Parent1+(1-beta2)*Parent2
    
    n = size(OldChrom,1)   # Number of individuals and chromosome length
    npairs = Int(floor(n/2))    # Number of pairs
       
    cross = (rand(npairs,1) .<= XOVR)    # Pairs to crossover
        
    NewChrom = deepcopy(OldChrom)
    
    
    #Threads.@threads 
    for i in 1:npairs
        pin = (i-1)*2+1
        if cross[i]!=0
            betas = rand(2,1)*(1+2*alpha).-(0.5+alpha)
            A=[betas[1] 1-betas[1]; 1-betas[2] betas[2]]
            NewChrom[pin:pin+1,:] = A*OldChrom[pin:pin+1,:]
        end
    end
    
    
    return NewChrom
    
    # Coerce points outside search space
    # aux = ones(n,1)
    # auxf1=aux*FieldDR(1,:)
    # auxf2=aux*FieldDR(2,:)
    # NewChrom = (NewChrom>auxf2).*auxf2+(NewChrom<auxf1).*auxf1+(NewChrom<=auxf2 & NewChrom>=auxf1).*NewChrom


end

"""
    mutbga(OldChrom::Array{Float64,2}, FieldDR::Array{Float64,2}; kwargs...)

  Mutation function
  Real coded mutation. 
  Mutation is produced adding a low random value
  # OldChrom: Initial population.
  # FieldChrom: Upper and lower bounds.
  # MutOpt: mutation options,
  #         MutOpt(1)=mutation probability (0 to 1).
  #         MutOpt(2)=compression of the mutation value (0 to 1).
  #         default MutOpt(1)=1/Nvar y MutOpt(2)=1
"""
function mutbga(OldChrom::Array{Float64,2}, FieldDR::Array{Float64,2}; kwargs...)
   pm, shr = get(kwargs, :Mutopt, Float64[1/size(FieldDR,2); 1])
   


    
    Nind = size(OldChrom,1)
    
    m1 = 0.5-(1-pm)*0.5
    m2 = 0.5+(1-pm)*0.5
    
    aux = rand(size(OldChrom)[1],size(OldChrom)[2])

    MutMx = (aux .> m2) .- (aux .< m1)

    range = [-1 1] * FieldDR * 0.5 * shr
    
    range = ones(Nind,1) * range
    

    
    
    index = findall(!iszero,MutMx)
    
    m = 20

    dim = length(index)
    temp = rand(m,dim)

    alpha = temp .< (1/m)
    
    xx    = 2.0 .^ (0:-1:(1-m))   
    aux2  = transpose(xx)*alpha
    
    delta = zeros(size(MutMx))
    delta[index] = aux2

    
    NewChrom = OldChrom .+ (MutMx.*range.*delta)

    # Coerce points outside bounds
    aux = ones(Nind,1)
    auxf1 = aux*transpose(FieldDR[1,:])
    auxf2 = aux*transpose(FieldDR[2,:])
    
    NewChrom = (NewChrom .>  auxf2) .* auxf2 .+ (NewChrom .<  auxf1) .* auxf1 .+ ((NewChrom .<= auxf2) .& (NewChrom .>= auxf1)) .* NewChrom
     
    return NewChrom
end





function gaevolution!(ObjV, gaDat::gaDatVar)

    # One generation 
    
    #ObjV  = vec(fill(Inf,gaDat.NIND,1))
    """
    tache = [([p_1,...p_25], M_1), ([p_26,...p_50], M_2),...] (-m_nthraeds))
    Threads.@threads for val in tache 
        multitrack(val...)
    end
    """
    #for i in 1:gaDat.NIND
    Threads.@threads for i in 1:gaDat.NIND
        @inbounds ObjV[i] = gaDat.Objfun(gaDat.Chrom[i,:])
    end
    

    gaDat.ObjV = ObjV
    
    
    # Best individual of the generation
    
    v,p = minimum(gaDat.ObjV),argmin(gaDat.ObjV)[1]
    
    if v <= gaDat.fxmin
        gaDat.xmin = gaDat.Chrom[p,:]
        gaDat.fxmin = v
    end
    
    # Optional additional task required by user
    gaiteration(gaDat)
    #@info "Iteration: $(gaDat.gen)"
    
    # Next generation
    
    # RANKING 
    FitnV = ranking(gaDat.ObjV,gaDat.rf)
    
    # SELECTION 
    # Stochastic Universal Sampling (SUS).
    SelCh = select(gaDat,FitnV)
    
    # CROSSOVER 
    # Uniform crossover.
    SelCh = lxov(SelCh,gaDat.Pc,gaDat.alfa)
    
    # MUTATION 
    Chrom = mutbga(SelCh,gaDat.FieldD,Mutopt=[gaDat.Pm 1])
    
    # Reinsert the best individual  
    Chrom[Int(round(gaDat.NIND/2)),:] = gaDat.xmin
    gaDat.Chrom=Chrom
    
end


function ga!(gaDat) 
    
    ################################################
    # Main loop
    ################################################
    # Generation counter
    
    gen=0
        
    # Initial population    
    
    # Real codification
    crtrp!(gaDat)
    
    # Individuals of gaDat.indini are randomly added in the initial population
    
    if !(isempty(gaDat.indini))
        nind0=size(gaDat.indini,1)
        posicion0=randperm(nind0)
        gaDat.Chrom[posicion0,:]=gaDat.indini
    end
    
    #gaDat.xmingen = zeros(gaDat.MAXGEN+1,gaDat.NVAR)
    #gaDat.fxmingen = zeros(gaDat.MAXGEN+1,1)

    ObjV  = vec(fill(Inf,gaDat.NIND,1))
    
    while (gaDat.gen < gaDat.MAXGEN)
        #global gen
        gaDat.gen = gen
        gaevolution!(ObjV,gaDat)  
        
        # Increase generation counter     
        gaDat.xmingen[gen+1,:] = gaDat.xmin
        gaDat.fxmingen[gen+1] = gaDat.fxmin
        gen=gen+1
        
    end
    
    ######################################################
    # End main loop
    ######################################################
    # Present final results
    
    garesults(gaDat)

    #@info "Objective function for fxmin: $(gaDat.fxmin)"
    #@info "xmin: $(gaDat.xmin)"


end
