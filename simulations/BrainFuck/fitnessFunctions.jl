function convertIntegerArrayToStringprogram(progIntegerArray::Array{Int64,1})
    symbols = [">","<","+","-",".",",","[","]"]
    progString= ""
    for i in 1:length(progIntegerArray)
        progString *= symbols[progIntegerArray[i]]
    end
    progString
end

#For this kind of program, we simply want the output String to be \"hard-coded\" in the source code, so we don't need any training. The length of the desired String will obviously influence drasticaly the speed of execution of the algorithm"
#"fitness calculation, best fitness is 0"

function fitnessStr(progInteger; ticksLim = 10000)
    #convert the program to String
    prog = convertIntegerArrayToStringprogram(progInteger)
	#expect_out = [72, 101, 108, 108, 111] # Hello
    #expect_out = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 10] # Hello World!
	expect_out = [104, 105] # hi
	
	prog_out,ticks_out = brainfuck(prog;ticks_lim=ticksLim)
	
	diff = 0
	
    if length(prog_out)<length(expect_out)
        pad_prog_out = append!(prog_out,zeros(Int64,length(expect_out)-length(prog_out)))
        pad_expect_out = expect_out
    elseif length(prog_out)>length(expect_out) #if the output is too long, we can simply cut the program
        pad_expect_out = expect_out
        pad_prog_out = prog_out[1:length(expect_out)]
    else
        pad_prog_out = prog_out
        pad_expect_out = expect_out
    end
    

    for i in eachindex(pad_prog_out)
        diff += abs(pad_prog_out[i]-pad_expect_out[i])
    end
	
    return Float64(diff)
    
end