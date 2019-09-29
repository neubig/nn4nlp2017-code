# # Log-linear Language Model

using Knet: Param, @diff, grad, params, value, KnetArray, logp, nll, progress
using Random: shuffle!

# We are using data from the Penn Treebank, which is already converted into an easy-to-use
# format with "&lt;unk&gt;" symbols. If we were using other data we would have to do
# pre-processing and consider how to choose unknown words, etc.

readdata(file)=[ w2i.(split(line)) for line in eachline(file) ]

# Before reading the training data, we initalize the word->id dictionary with sentence
# separator "&lt;s&gt;" and unknown word "&lt;unk&gt;" symbols.

wdict = Dict()
w2i(x) = get!(wdict, x, 1+length(wdict)) # insert key with value len+1 if not found
S = w2i("<s>")
UNK = w2i("<unk>")

# Load the training data and peek at the first instance:

trn = readdata("../data/ptb/train.txt")
first(trn)'

# Here is the reconstructed string of the first sentence:

wstring = Array{String}(undef,length(wdict))
for (str,id) in wdict; wstring[id] = str; end
i2w(i) = wstring[i]
join(i2w.(rand(trn)), " ")

# Before reading the dev/test data, we change the word->id function to return UNK for
# unknown words.

w2i(x) = get(wdict, x, UNK)     # return UNK if x is not found

# Load the dev/test data and print the number of instances:

dev = readdata("../data/ptb/valid.txt")
tst = readdata("../data/ptb/test.txt")
length.((trn, dev, tst))

# Use KnetArray to initialize parameters on GPU, use Array to initialize on CPU:

##param(dims...) = Param(Array(0.01f0 * randn(Float32, dims...)))
param(dims...) = Param(KnetArray(0.01f0 * randn(Float32, dims...)))

# Initialize the parameters of the loglin-lm model as global variables W and b.

nwords = length(wdict)
N = 2  # The length of the n-gram
W = [ param(nwords, nwords) for i in 1:N ]
b = param(nwords)

# Here is the loss function for a whole sentence:

function loss(sent)
    slen = length(sent)
    input = [ repeat([S],N); sent ]
    scores = b
    for i in 1:N
        scores = scores .+ W[i][:,input[i:i+slen]] # @size scores (V,slen+1)
    end
    nll(scores, [sent; [S]])
end

# Here is the SGD training loop:

function train(data=trn; nepochs = 10, lr = 0.01f0)
    for epoch in 1:nepochs
        shuffle!(data)
        lastloss = 0
        for s in progress(x->lastloss, data)
            ∇loss = @diff loss(s)
            lastloss = value(∇loss)
            for p in params(∇loss)
                p .= p - lr * grad(∇loss, p)
            end
        end
    end
end
