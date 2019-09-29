SRC := loglin-lm.jl

TGT := loglin-lm.ipynb

all: ${TGT}

%.ipynb: %.jl
	julia -e 'using Literate; Literate.notebook("$<", ".", execute=false)'
