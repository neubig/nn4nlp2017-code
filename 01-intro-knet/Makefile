SRC := bow.jl cbow.jl deep-cbow.jl

TGT := bow.ipynb cbow.ipynb deep-cbow.ipynb

all: bow.ipynb cbow.ipynb deep-cbow.ipynb

%.ipynb: %.jl
	julia -e 'using Literate; Literate.notebook("$<", ".", execute=false)'
