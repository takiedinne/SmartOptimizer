### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 888f3d44-7d9a-11eb-35a4-85d7fa64d6e5
module Brainfuck

using Match

export BFProg, find_matching_bracket, find_matching_bracket_reverse, brainfuck, filter_bad_candidate, generate_rand_prog, purify_code



"find the matching bracket in a string, function needed in the interpreter to find the matching ']' of a '['"
function find_matching_bracket(str::String)
	counter_open = 1
	counter_close = 0
	for i in eachindex(str)
		if str[i] == '['
			counter_open += 1
		elseif str[i] == ']'
			counter_close += 1
		end
		
		if counter_open==counter_close
			return i
		end
	end
end

function find_matching_bracket_reverse(str::String)
	counter_close = 1
	counter_open = 0
	for i in reverse(1:length(str))
		if str[i] == '['
			counter_open += 1
		elseif str[i] == ']'
			counter_close += 1
		end
		
		if counter_open==counter_close
			return i
		end
	end
end

"Brainfuck interpreter, using a @match macro"
function brainfuck(prog::String, input::Array{UInt8,1}; memsize::Int64=500, ticks_lim::Int64=10000)
	out = UInt8[]
    
	# Read program and filter symbols
	symbols = ['>','<','+','-','.',',','[',']']
	code = filter(x -> in(x, symbols), prog)

	# Memory of the program
	memory = zeros(UInt8, memsize) # Memory in Int64

	# Stack for loops
	stack = Int64[]
	ptr = 1                 # Memory pointer
	instr = 1               # Instruction pointer
	
	ticks = 0 # ticks counter for timeout and fitness calculation

	# Run the program
	while instr <= length(code) && ticks <= ticks_lim
		# Wrapping the memory
		if ptr > memsize
			ptr = ptr - memsize
		end
		if ptr <= 0
			ptr = ptr + memsize
		end
		
		# Instruction matching
		@match code[instr] begin
			'>' => (ptr += 1)
			'<' => (ptr -= 1)
			'+' => (if memory[ptr] + 1 < 256
					memory[ptr] += 1
				else
					memory[ptr] = 0
				end)
			'-' => (if memory[ptr] - 1 >= 0
					memory[ptr] -= 1
				else
					memory[ptr] = 255
				end)
			'.' => push!(out,memory[ptr]) # Decimal OUTPUT (Int64)
			',' => (if length(input) != 0
					memory[ptr] = popfirst!(input)
				else
					memory[ptr] = 0
				end)
			'[' => (if memory[ptr] == 0
					if find_matching_bracket(code[instr+1:end]) !== nothing
						instr += find_matching_bracket(code[instr+1:end])
					end
				else
					push!(stack, instr)
				end)
			']' => (if memory[ptr] != 0
					if length(stack) != 0
						instr = pop!(stack) - 1
					end
				else
					if length(stack) != 0
						pop!(stack)
					end
				end)      
		end
		instr += 1
		ticks += 1
	end
	return out,ticks
end

function brainfuck(prog::String; memsize::Int64=500, ticks_lim::Int64=10000)
	brainfuck(prog, Array{UInt8,1}())
end

"function filtering bad candidates for a brainfuck program on basis of their code. this function asserts brackets matching"
function filter_bad_candidate(prog::String)

	for i in 1:length(collect(prog))
		if length(findall(collect(prog[1:i]) .== ']')) > length(findall(collect(prog[1:i]) .== '['))
			return "bad"
		end
	end

	for i in 1:length(collect(reverse(prog)))
		if length(findall(collect(reverse(prog)[1:i]) .== '[')) > length(findall(collect(reverse(prog)[1:i]) .== ']'))
			return "bad"
		end
	end

	return "good"

	#TBC

end
	
	
"random individuals generation, programs (individuals) are stored as strings"
function generate_rand_prog(max_size::Int64)
	symbols = ['>','<','+','-','.',',','[',']']
	
	size = rand(5:max_size)
	code = rand(symbols,size)
	
	return join(code)
end

"struct for the usage of brainfuck programs in a genetic algorithm, storing the program and its fitness"
mutable struct BFProg
	program
	fitness::Int64
	
	function BFProg(maxProgSize::Int64)
		return new(generate_rand_prog(maxProgSize),10^10)
	end
	
	function BFProg(program,fitness::Int64)
		return new(program,fitness)
	end
end

function filter_code(bf_prog::String)
	pure_bf = bf_prog
	pure_bf = replace(pure_bf,"+-"=>"")
	pure_bf = replace(pure_bf,"-+"=>"")
	pure_bf = replace(pure_bf,"<>"=>"")
	pure_bf = replace(pure_bf,"><"=>"")
	return pure_bf
end

"function to eliminate useless characters in a brainfuck program"
function purify_code(bf_prog::String)
	global old = ""
	global new1 = bf_prog
	while new1 != old
		global new1
		global old
		old = new1
		new1 = filter_code(old)
	end
	indexToDelete = Int64[]
	for i in eachindex(new1)
		if new1[i] == '['
			if find_matching_bracket(new1[i+1:end]) === nothing
				append!(indexToDelete,[i])
			end
		end
	end
	
	for i in eachindex(new1)
		if new1[i] == ']'
			if i == 1
				append!(indexToDelete,[i])
			elseif find_matching_bracket_reverse(new1[1:i-1]) === nothing
				append!(indexToDelete,[i])
			end
		end
	end
	new1 = collect(new1)	
	deleteat!(new1,sort(indexToDelete))
	return join(new1)
end

end	


# ╔═╡ Cell order:
# ╠═888f3d44-7d9a-11eb-35a4-85d7fa64d6e5
