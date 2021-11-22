using SimJulia
using ResumableFunctions
using QuadGK
using Distributions
using Plots
using XLSX



mutable struct Projectile
    position::Float64
    velocity::Float64
    mass::Float64
    calibre::Float64
end

mutable struct Fragment
    position::Float64
    velocity::Float64
    mass::Float64
    calibre::Float64
end

mutable struct Target #momenteel niet gebruikt
    θ::Float64 # angle w.r.t. artificial north
    ρ::Float64 # distance to own ship
    v::Float64 # speed of the target
    d::Float64 # diameter of the target
    A::Float64 #target area
    Type::Int64 #0: UAV, 1: Missile, 2:FIAC/LFA

end

mutable struct Sim_results
    veject::Array{Float64,1}
    pvelocity::Array{Float64,1}
    coneangle::Array{Float64,1}
    alethal::Array{Float64,1}
    rlethal::Array{Float64,1}
    nbrfragment::Array{Float64,1}
    vifragment::Array{Float64,1}
    vffragment::Array{Float64,1}
    sshp::Array{Float64,1}
    pkfragment::Array{Float64,1}
    pkprojectile::Array{Float64,1}
    pkh::Array{Float64,1}
    rounds::Float64
    missed::Float64
    ekin::Float64
    pk::Float64
end

#################################################################
#### Input parameters ##########################################
################################################################

#    number_of_fragments =83#135#5#10#20#50#100#135 #this is the number of fragments in the cone (maybe a fraction of the total number of fragments)
fragment_diameter = 0.004 # m

fragments_mass = 0.2 #kilograms  inputs
#target_presented_area = 20 #m2 input
#all fragments have the same velocity and the same mass
#detonation_distance = 1.0 #10# m the projectile points towards the projectile

angle_target = 0.0 #degrees
distance_target = 0.0 # meters ce paramètre n'est pas utilisé pour l'instant
speed_target = 0.0 #m/s
diameter_target = 1.0#1.0 #m
area_target = pi * (diameter_target / 2)^2

#α = 12 #degrees cone angle

const cd0_fragment = 1.2
const mass = 0.423#0.234 kilograms
const calibre = 0.030 # meters
#angular_velocity = 10^4/2 #rad/sec
const density = 1.2244 #kg/m3
#velocity = 250.0#340.0#250.0 #1200.0 # m/s
E_ref = 20 #J in the master thesis it was fixed at 15 jouls
#engagement_distance = 200.0#500.0#2000.0#500.0# m
const σball = 0.001
const σatm = 1.0#0.001
const HIT_PROBABILITY = 0.95
const sim_burst = 0 #0: schot-per-schot; 1: max burst size; 2: Intelligent burst
const MAX_BURST_SIZE = 150 # rounds
detonation_afstand = 5.0#1.00 m
spin = 1 #0:Spin stabilized projectile ; 1: explosive charge to eject fragment
v_eject = 95.0#20.0 #m/s
delta_t = 0.001 # s precision for trajectory computation
#delta_x = 2.0 #m distance to detonate from target

v_char = 2800.0 #m/s sqrt(2E) for HMX in gurney equation
#explosif_mass = 0.001 #kg

#############################################
##les paramètres pour l'intelligent burst####
#############################################
v_min = 250.0 # m/s
RoF = 100#75 # Hz
v_max = 1000.0 #m/s

############################################################################
############### end of input parameters ####################################
############################################################################


############################################################################
############### functions definition ######################################
###########################################################################
function gurney()
    v = v_char * (fragments_mass / explosif_mass + 1 / 2)^(-1 / 2) 
    
end

function opening_angle(p::Projectile, vp::Float64)
    if vp < p.velocity

        #v_rot = v_rot_ini/velocity*p.velocity
        α = asind(vp / p.velocity) * 2
    else
        α = asind(p.velocity / vp) * 2
    end
    #rad2deg(α)
end
function opening_angle(p::Projectile)

    #v_rot = v_rot_ini/velocity*p.velocity
    α = asind(v_eject / p.velocity) * 2
    #rad2deg(α)
end

@resumable function detonation(sim::Simulation, p::Projectile, flight_time::Float64, target::Target, f::Fragment, vp::Float64, d::Float64)
    #    function detonation(p::Projectile)
    #if spin ==0
    α = opening_angle(p, vp)
    #println(" angle" , " " , α)
    push!(res.coneangle, α)
    #elseif spin == 1
    #    α = opening_angle(p)
    #    println(" angle" , " " , α)
    #end

    #n_disp, d = number_of_hit(target, p, flight_time, α, f)
    n_disp, r_lethal = number_of_hit(target, p, flight_time, α, f, d)
    #println("fragments", " ", n_disp," ", " distance"," ", d)
    push!(res.nbrfragment, n_disp)
    f.mass = fragments_mass / number_of_fragments
    #f.velocity = sqrt(p.velocity^2+(p.velocity*sind(α/2))^2)

    #n_disp, d
    n_disp, r_lethal
end

#@resumable function number_of_effective_fragment(sim::Simulation)
function number_of_effective_fragment(d::Float64, α::Float64, target::Target)
    solid_angle = α / 57.3
    #projection_area = d^2*solid_angle
    r = d * tand(α / 2) #lethal radius

    projection_area = pi * r^2 #lethal area
    #println("lethal_area", " ", projection_area)
    push!(res.alethal, projection_area)

    if target.A < projection_area
        n = target.A / projection_area * number_of_fragments
    else
        n = number_of_fragments
    end
    return n, r

end


function time_fuse(p::Projectile, f::Fragment, α::Float64)
    solid_angle = α / 57.3
    #projection_area = detonation_distance^2*solid_angle

    detonation_distance = sqrt(lethal_area / solid_angle)
end

function detonation_distance(f::Fragment, r::Float64, α::Float64)
    lethal_number = lethal_number_of_fragment(f)#/r
    #r = dispersion_factor(target.A, p.position, flight_time)
    lethal_area = target.A / lethal_number * number_of_fragments
    solid_angle = α / 57.3
    detonation_distance = sqrt(lethal_area / solid_angle)
    detonation_distance, lethal_number
end

#@resumable function kill_probability(f::Fragment)
function kill_probability(f::Fragment)
    E_impact = 1 / 2 * f.mass * f.velocity^2
    Pkh = 1 - exp(-E_impact / E_ref)
end



#@resumable function trajectory_projectile(sim::Simulation, p::Projectile, dt::Float64)
function trajectory(p::Projectile, dt::Float64)
    ma = p.velocity / 340.0
    if ma > 4.0
        cd0 = 0.647 * ma^(-0.5337)
    else
        cd0 = 0.0065 * ma^2 - 0.0965 * ma + 0.5901
    end
    drag = -0.125 * density * pi * p.calibre^2 * cd0 * p.velocity^2
    acceleration = drag / p.mass
    p.velocity += acceleration * dt
    p.position += p.velocity * dt
end

#@resumable function trajectory_fragment(sim::Simulation, f::Fragment, dt::Float64)
function trajectory(f::Fragment, dt::Float64)
    drag = -0.125 * density * pi * f.calibre^2 * cd0_fragment * f.velocity^2
    acceleration = drag / f.mass
    f.velocity += acceleration * dt
    f.position += f.velocity * dt
end

@resumable function projectile_flight(sim::Simulation, p::Projectile, ρ::Float64, dt::Float64, dx::Float64)
    #function flight(p::Projectile, ρ::Float64, dt::Float64)
    t = 0.0
    old_position = p.position
    old_velocity = p.velocity

    while ρ - dx > p.position
        old_position = p.position
        old_velocity = p.velocity
        t += dt
        #@process trajectory_projectile(sim, p, dt)
        trajectory(p, dt)
    end
    flight_time = t - dt + dt * (ρ - old_position) / (p.position - old_position)
    #println(" position 1", " ", p.position)
    p.position = old_position
    p.velocity = old_velocity + (p.velocity - old_velocity) * (flight_time - t + dt) / dt

    flight_time


end

@resumable function fragment_flight(sim::Simulation, f::Fragment, ρ::Float64, dt::Float64)
    #function flight(f::Fragment, ρ::Float64, dt::Float64)
    t = 0.0
    old_position = f.position
    old_velocity = f.velocity

    while ρ > f.position
        old_position = f.position
        old_velocity = f.velocity
        t += dt
        #@process trajectory_fragment(sim, f, dt)
        trajectory(f, dt)
    end
    flight_time = t - dt + dt * (ρ - old_position) / (f.position - old_position)
    #f.position = old_position
    f.velocity = old_velocity + (f.velocity - old_velocity) * (flight_time - t + dt) / dt
    flight_time

end

function σtot(ρ::Float64, flight_time::Float64)
    sqrt((ρ * σball)^2 + (flight_time * σatm)^2)
end

function dispersion_factor(size::Float64, D::Float64, flight_t::Float64, target::Target, r::Float64)
    #flight_t = flight_time(Projectile(0.0, velocity, mass, calibre), D, 0.1)[1]
    Stot = σtot(D, flight_t)
    #y = (x)->x/Stot^2*exp(-0.5*x^2/Stot^2)
    #g=quadgk(y,0,0.5*size)[1]
    r = abs(rand(Truncated(Normal(0, Stot / (1 / 2 * target.d + r)), -1, 1)))
    #r = abs(rand(Truncated(Normal(0,Stot),-1,1)))
    #return g
end

function SSHP(size::Float64, D::Float64, flight_t::Float64)
    #flight_t = flight_time(Projectile(0.0, velocity, mass, calibre), D, 0.1)[1]
    Stot = σtot(D, flight_t)
    y = (x) -> x / Stot^2 * exp(-0.5 * x^2 / Stot^2)
    g = quadgk(y, 0, 0.5 * size)[1]
    return g
end

#@resumable function toto(sim::Simulation)
#    @yield timeout(sim, 5)
#end
function lethal_number_of_fragment(f::Fragment)
    n = 0
    Pkh = 1.0
    while (1 - Pkh) < HIT_PROBABILITY && n < number_of_fragments
        Pkhi = kill_probability(f)
        Pkh = Pkh * (1 - Pkhi)
        n = n + 1
    end
    n
end

@resumable function intelligent_burst(sim::Simulation, tf::Float64, D::Float64, vmax::Float64, t::Float64)
    #t = flight_time_lu(Projectile_lu(0.0, vmin, mass, calibre), D, 0.1)[1]
    #println("tijd i", " ", tf)
    told = Inf
    v = v_min
    #v_guess = v+Δv
    #    println("v initiale", " ", v_guess)
    Δv = 10
    #    p_guess = Projectile(0.0, v_guess, mass, calibre)
    #    tf_guess = @yield @process projectile_flight(sim, p_guess, engagement_distance, 0.01)


    while t > tf
        v = v + Δv
        #println("v initiale", " ", v)
        if v > v_max
            #println("Flight time too low, can't raise muzzle velocity: tf = $tf, t = $t, d = $D, v = $(D/tf)")
            v = 0.0
            tf = 0.0
            t = 0.0
            break
        end
        told = t
        #t = flight_time_lu(Projectile_lu(0.0, v, mass, calibre), D, 0.1)[1]
        #p_guess = Projectile(0.0, v_guess, mass, calibre)
        t = @yield @process projectile_flight(sim, Projectile(0.0, v, mass, calibre), engagement_distance, 0.01)
    end
    v = v + 10 * (tf - t) / (t - told)
    #return v
    v
end

function fragment_vel(p::Projectile, vmuzzle::Float64)
    if spin == 1
        v = sqrt(p.velocity^2 + v_eject^2)
    else
        v_rot = v_rot_ini / vmuzzle * p.velocity
        v = sqrt(p.velocity^2 + v_rot^2)
    end
    v
end

function perpendicular_velocity(p::Projectile, vmuzzle::Float64)
    if spin == 1
        #vp = v_eject
        vp = gurney()
    else
        vp = v_rot_ini / vmuzzle * p.velocity
    end
    vp
end

@resumable function burst_length(sim::Simulation, target::Target)
    Pnhit = 1.0
    burst = 0
    miss = 0
    D = target.ρ
    Pkn = 1.0
    Pkhn = 1.0
    Ekinb = 0.0

    if sim_burst == 0
        VEND = Float64[]
        while (1 - Pkhn) < HIT_PROBABILITY && burst < MAX_BURST_SIZE
            #sshp, vend = SSHP(velocity_max, target, D)
            p = Projectile(0.0, velocity, mass, calibre)
            #println("toto1")
            flight_time = @yield @process projectile_flight(sim, p, engagement_distance, delta_t, delta_x)
            #println("projectile velocity", " ", p.velocity)
            #println("projectile position", " ", p.position)
            push!(res.pvelocity, p.velocity)

            Std = σtot(p.position, flight_time)
            #r = abs(rand(Truncated(Normal(0,Std),-1,1)))
            r = rand(Normal(0, Std))
            p.position = p.position + r
            #println("updated projectile position", " ", p.position)

            if p.position < engagement_distance
                #println("target missed")
                #break

                #push!(VEND, p.velocity)
                #v_rot = v_rot_ini/velocity*p.velocity
                #fragment_velocity = sqrt(p.velocity^2 + v_rot^2 )
                vp = perpendicular_velocity(p, velocity)
                #println("fragments perpendicular velocity", " ", vp)
                #push!(res.veject,vp)
                fragment_velocity = sqrt(p.velocity^2 + vp^2)
                #fragment_velocity = fragment_vel(p, velocity)

                f = Fragment(0, fragment_velocity, fragment_mass, fragment_diameter)
                #detonation_param = @yield @process detonation(sim, p, flight_time, target, f, v_rot)
                d = engagement_distance - p.position
                #println("detonation distance", " ", d)
                det_param = @yield @process detonation(sim, p, flight_time, target, f, vp, d)
                n_disp = det_param[1]
                r_lethal = det_param[2]
                #println("fragment velocity", " ", f.velocity, " ", "fragment prosition", " ", f.position)
                push!(res.vifragment, f.velocity)
                #d = detonation_param[2]
                #println("lethal radius", " ", r_lethal)
                push!(res.rlethal, r_lethal)
                size = target.d + r_lethal * 2

                Phiti = SSHP(size, p.position, flight_time)
                #println("sshp", " ", Phiti)
                push!(res.sshp, Phiti)
                Pnhit = Pnhit * (1 - Phiti)

                @yield @process fragment_flight(sim, f, d, 0.01)

                #println("fragment velocity", " ", f.velocity)
                push!(res.vffragment, f.velocity)

                Pkfi = kill_probability(f)
                #println("pki for fragment", " ", Pkfi )
                push!(res.pkfragment, Pkfi)
                #n_disp = detonation_param[1]
                #println("number of fragments", " ", n_disp)
                Pki = 1 - (1 - Pkfi)^n_disp
                Pkhi = Pki * Phiti
                #println(" pki for projectile", " ", Pki)
                push!(res.pkprojectile, Pki)
                #println(" pkhi for projectile", " ", Pkhi)
                push!(res.pkh, Pkhi)



                Pkn = Pkn * (1 - Pki)
                #println(" pkill" , " ", 1-Pkn)
                Pkhn = Pkhn * (1 - Pkhi)

                #println(" pkill if hit", " ", 1-Pkhn)

            else
                miss = miss + 1
                #println("target missed")
            end
            burst = burst + 1
            #println("burst", " ", burst)
            #println("missed", " ", miss, " ", "times")
            #D=D-target.v * cosd(target.δ)/RATE_OF_FIRE
            #if D<target.Rmin
            #    break
            #end
        end
        Pktot = (1 - Pkhn)
        Ekinb = burst * 0.5 * mass * (velocity)^2
        #burst, Pktot, Ekinb, mean(VEND)
        res.rounds = burst
        res.missed = miss
        res.pk = Pktot
        res.ekin = Ekinb
        push!(missed, miss)
        burst, Pktot, Ekinb
    elseif sim_burst == 1
        #VEND = Float64[]
        #sshp, vend = SSHP(velocity_max, target, D)
        #println("toto1")
        p = Projectile(0.0, velocity, mass, calibre)
        flight_time = @yield @process projectile_flight(sim, p, engagement_distance, delta_t, delta_x)
        #println("projectile velocity", " ", p.velocity)
        #println("projectile position", " ", p.position)
        Std = σtot(p.position, flight_time)
        #r = abs(rand(Truncated(Normal(0,Std),-1,1)))
        r = rand(Normal(0, Std))
        #p.position = p.position + r
        #println("updated projectile position", " ", p.position)

        #if p.position > engagement_distance
        #    println("target missed")
        #break
        #end
        #push!(VEND, p.velocity)
        #v_rot = v_rot_ini/velocity*p.velocity
        #fragment_velocity = sqrt(p.velocity^2 + v_rot^2 )
        vp = perpendicular_velocity(p, velocity)
        fragment_velocity = sqrt(p.velocity^2 + vp^2)
        #fragment_velocity = fragment_vel(p, velocity)
        f = Fragment(0, fragment_velocity, fragment_mass, fragment_diameter)
        #detonation_param = @yield @process detonation(sim, p, flight_time, target, f, v_rot)
        d = engagement_distance - p.position
        #println("detonation distance", " ", d)
        f_flight = @yield @process fragment_flight(sim, f, d, delta_t)
        #println(" ")
        det_param = @yield @process detonation(sim, p, flight_time, target, f, vp, d)
        n_disp = det_param[1]
        r_lethal = det_param[2]
        #println("fragment velocity", " ", f.velocity, " ", "fragment prosition", " ", f.position)
        #d = detonation_param[2]
        #println("lethal radius", " ", r_lethal)
        size = target.d + r_lethal * 2

        Phiti = SSHP(size, p.position, flight_time)
        #println("sshp", " ", Phiti)
        #println("fragment velocity", " ", f.velocity)
        Pkfi = kill_probability(f)
        #println("pki for fragment", " ", Pkfi )
        #n_disp = detonation_param[1]
        #println("number of fragments", " ", n_disp)
        Pki = 1 - (1 - Pkfi)^n_disp
        Pkhi = Pki * Phiti
        #println(" pki for projectile", " ", Pki)
        #println(" pkhi for projectile", " ", Pkhi)
        #push!(VEND, vend)
        if Pkhi > HIT_PROBABILITY
            burst = 1
        else
            burst = Int(ceil(log(1 - HIT_PROBABILITY) / log(1 - Pkhi)))
        end
        if burst > MAX_BURST_SIZE
            burst = MAX_BURST_SIZE
        end
        Pktot = 1 - (1 - Pkhi)^burst
        #Phit = 1-(1-sshp)^burst
        #Ekinb = burst * 0.5*mass*(velocity_max)^2
        Ekinb = burst * 0.5 * mass * (velocity)^2
        #burst, Phit, Ekinb, mean(VEND)
        burst, Pktot, Ekinb
    elseif sim_burst == 2
        # while (1-Pkhn) < HIT_PROBABILITY && burst<MAX_BURST_SIZE
        p = Projectile(0.0, v_min, mass, calibre)
        #flight_time = @yield @process projectile_flight(sim, p, engagement_distance, 0.01)
        Ttot = @yield @process projectile_flight(sim, p, engagement_distance, delta_t, delta_x)

        #burst = 3
        #println(Ttot)
        #size = target.d
        i = 1

        #for i in 1:burst
        while (1 - Pkhn) < HIT_PROBABILITY && burst < MAX_BURST_SIZE
            #while (1-Pkn) < HIT_PROBABILITY && burst<MAX_BURST_SIZE
            #    println(i)
            #end
            Tfi = Ttot - (i - 1) / RoF
            #    println("tfi", " ", Tfi)
            vi = @yield @process intelligent_burst(sim, Tfi, engagement_distance, v_max, Ttot)
            #    println("velocity: $vi")
            if vi == 0.0
                #        println("velocity te hoog")
                #burst = burst-1
                #@goto End
                break
            end
            # sshp, vend = SSHP_lu(Projectile_lu(0.0, velocity, mass, calibre), Target_lu(D, v_target, L_target, D_target, δ), R_let, σball, σatm, Verror)

            pi = Projectile(0.0, vi, mass, calibre)
            tfi = @yield @process projectile_flight(sim, pi, engagement_distance, delta_t, delta_x)
            #    println("projectile velocity", " ", pi.velocity)
            #    println("projectile position", " ", pi.position)

            Std = σtot(pi.position, tfi)
            #r = abs(rand(Truncated(Normal(0,Std),-1,1)))
            r = rand(Normal(0, Std))
            pi.position = pi.position + r
            #    println("updated projectile position", " ", pi.position)
            #v_roti = v_rot_ini/velocity*pi.velocity
            #fragment_velocity = sqrt(pi.velocity^2 + v_roti^2 )
            if pi.position < engagement_distance
                vp = perpendicular_velocity(pi, vi)
                fragment_velocity = sqrt(pi.velocity^2 + vp^2)
                #fragment_velocity = fragment_vel(p, vi)
                fi = Fragment(0, fragment_velocity, fragment_mass, fragment_diameter)
                d = engagement_distance - pi.position
                #println("detonation distance", " ", d)
                det_param = @yield @process detonation(sim, pi, tfi, target, fi, vp, d)
                n_dispi = det_param[1]
                r_lethal = det_param[2]
                #println("fragment velocity", " ", fi.velocity, " ", "fragment prosition", " ", fi.position)
                #    println("lethal radius", " ", r_lethal)
                size = target.d + r_lethal * 2

                Phiti = SSHP(size, pi.position, tfi)
                #println("sshp: $sshp, vend: $vend")
                #    println("flight time"," ", tfi)
                #    println("Phit", " ",Phiti )
                #                push!(VEND, vend)
                Ekinb = Ekinb + 0.5 * mass * (vi)^2
                Pnhit = Pnhit * (1 - Phiti)
                @yield @process fragment_flight(sim, fi, d, 0.01)
                #     println("fragment velocity", " ", fi.velocity)
                Pkfi = kill_probability(fi)
                #     println("pki for fragment", " ", Pkfi )
                #     println("number of fragments", " ", n_dispi)
                Pki = 1 - (1 - Pkfi)^n_dispi
                Pkhi = Pki * Phiti
                #     println(" pki for projectile", " ", Pki)
                #     println(" pkhi for projectile", " ", Pkhi)
                Pkn = Pkn * (1 - Pki)
                #     println(" pkill" , " ", 1-Pkn)
                Pkhn = Pkhn * (1 - Pkhi)

                #     println(" pkill if hit", " ", 1-Pkhn)
            else
                miss = miss + 1
                #     println("target missed")
            end
            #                Pnhitold = Pnhit
            #                Ekinbold = Ekinb
            #                velocityold = velocity
            burst = burst + 1
            i = i + 1
            #     println("burst", " ", burst)
            #     println("missed", " ", miss, " ", "times")
        end
        Pktot = (1 - Pkhn)
        #Pktot = 1-Pkn
        #Ekinb = burst * 0.5*mass*(velocity)^2
        #burst, Pktot, Ekinb, mean(VEND)
        #@label End
        burst, Pktot, Ekinb
    end
end

function number_of_hit(target::Target, p::Projectile, flight_time::Float64, α::Float64, f::Fragment)
    r = dispersion_factor(target.A, p.position, flight_time)
    d, n = detonation_distance(f, r, α)
    #n = number_of_effective_fragment(d, α)
    #println("orig number", " ", n)
    #n_disp = floor(n* (1-r))
    n_disp = n
    n_disp, d
end

function number_of_hit(target::Target, p::Projectile, flight_time::Float64, α::Float64, f::Fragment, d::Float64)

    #d, n = detonation_distance(f, r, α)
    n, r_lethal = number_of_effective_fragment(d, α, target)
    r = dispersion_factor(target.A, p.position, flight_time, target, r_lethal)
    #println("orig number", " ", n)
    n_disp = floor(n * (1 - r))
    #n_disp = floor(n)
    #n_disp, d
    n_disp, r_lethal
end



@resumable function main(sim::Simulation, target::Target)

    result = @yield @process burst_length(sim, target)
    #output = result[3]
    #res.veject = result[4]
    push!(output, result[3])
    push!(rounds, result[1])
    push!(Pk, result[2])
    #@info result# Pktot
    #push!(output,result[3])
end

##############################################################################
################# end of function definition #################################
##############################################################################

res = Sim_results([], [], [], [], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0, 0.0)
rounds = Float64[]
output = Float64[]
input_value = Float64[]
missed = Float64[]
Pk = Float64[]
min = 0.001
max = 0.04

function configuration(input_1::Float64, input_2::Float64, input_3::Float64, input_4::Float64)

    global number_of_fragments = input_1
    global delta_x = input_2
    global velocity = input_3
    global explosif_mass = input_4

    global engagement_distance = 200.0

    fragments = Fragment[]
    #target = UAV(angle_target, distance_target, speed_target, size_target,true)
    global target = Target(angle_target, distance_target, speed_target, diameter_target, area_target, 0)

    global output = Float64[]
    #output = 0.0
    global fragment_mass = fragments_mass / number_of_fragments
    
    #v_rot_ini = angular_velocity*calibre
    global v_rot_ini = velocity / 10
    #velocity = sqrt(p.velocity^2+(p.velocity*sind(α/2))^2)
    #fragment = Fragment(0, p.velocity, fragment_mass, fragment_diameter)
    sim = Simulation()
    @process main(sim, target)
    run(sim)
    #push!(input_value,i)
    return output[1]
end
