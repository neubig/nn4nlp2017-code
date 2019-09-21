# # Bag of Words (BOW) model

using Knet: Param, @diff, grad, params, KnetArray
using Random: shuffle!

# Data has one instance per line, class and sentence separated by `|||`, for example:
#
#     4 ||| A warm , funny , engaging film .
#
# `readdata(file)` reads the file and returns an array of (x,y) pairs where x is a sequence
# of word ids for a sentence, and y is a tag id. Uses `w2i` and `t2i` functions to turn
# words and tags to integers.

function readdata(file)
    data = []
    for line in eachline(file)
        tag, sentence = split(strip(lowercase(line)), " ||| ")
        wordids = w2i.(split(sentence))
        tagid = t2i(tag)
        push!(data, (wordids, tagid))
    end
    return data
end

# Before reading the training data, we initalize the word->id and tag->id dictionaries and
# functions and the unknown word tag `"<unk>"`.

wdict = Dict()
tdict = Dict()
w2i(x) = get!(wdict, x, 1+length(wdict))
t2i(x) = get!(tdict, x, 1+length(tdict))
UNK = w2i("<unk>")

# Load the training data and peek at the first instance:

trn = readdata("../data/classes/train.txt")
first(trn)

# Before reading the dev/test data, we change the word->id function to return UNK for
# unknown words and tag->id function to error for unknown tags.

w2i(x) = get(wdict, x, UNK)     # unk if not found
t2i(x) = tdict[x]               # error if not found

# Load the dev/test data and print the number of instances:

dev = readdata("../data/classes/dev.txt")
tst = readdata("../data/classes/test.txt")
length.((trn, dev, tst))

# Initialize the parameters of the BOW model as global variables W and b.

param(dims...) = Param(KnetArray(0.01f0 * randn(Float32, dims...)))
nwords = length(wdict)
ntags = length(tdict)
W = param(ntags, nwords)
b = param(ntags)

# Here is the prediction function for the BOW model:

pred(words) = b .+ sum(W[:,words], dims=2)

# Here is the loss function for the BOW model:

function loss(words, tag)
    scores = pred(words)
    logprobs = scores .- log(sum(exp.(scores)))
    -logprobs[tag]
end

# We use the following to report accuracy during training:

accuracy(data) = sum(argmax(pred(x))[1] == y for (x,y) in data) / length(data)

# Here is the SGD training loop:

function train(; nepochs = 10, lr = 0.01)
    for epoch in 1:nepochs
        shuffle!(trn)
        for (x,y) in trn
            ∇loss = @diff loss(x,y)
            for p in params(∇loss)
                p .= p - lr * grad(∇loss, p)
            end
        end
        println((epoch = epoch, trn = accuracy(trn), dev = accuracy(dev)))
    end
end
