# # Convolutional Sentiment Classification Network

using Knet
using Random, Statistics, Printf

# We are using the data from Stanford Sentiment Treebank dataset
# without tree information. First, we initialize our word->id and
# tag->id collections and insert padding word "&lt;pad&gt;" and
# unknown word "&lt;unk&gt;" symbols into word->id collection.

wdict, tdict = Dict(), Dict()
w2i(x) = get!(wdict, x, 1+length(wdict))
t2i(x) = get!(tdict, x, 1+length(tdict))
PAD = w2i("<pad>")
UNK = w2i("<unk>")

# In the data files, each line consists of sentiment and sentence
# information separated by `|||`.

function readdata(file)
    instances = []
    for line in eachline(file)
        y, x = split(line, " ||| ")
        y, x = t2i(y), w2i.(split(x))
        length(x) < 3 && continue
        push!(instances, (x,[y]))
    end
    return instances
end

# After reading training data, we redefine ```w2i``` procedure to
# avoid inserting new words into our vocabulary collection and then
# read validation data.

trn = readdata("../data/classes/train.txt")
w2i(x) = get(wdict, x, UNK)
t2i(x) = tdict[x]
nwords, ntags = length(wdict), length(tdict)
dev = readdata("../data/classes/test.txt")

# We begin developing convolutional sentiment classification model.
# Our model is a stack of five consecutive operations: word embeddings,
# 1-dimensional convolution, max-pooling, ReLU activation and linear
# prediction layer. First, we define our network,

mutable struct CNN
    embedding
    conv1d
    linear
end

# Then, we implement the forward propagation and loss calculation,

function (model::CNN)(x)
    emb = model.embedding(x)
    T, E = size(emb); B = 1
    emb = reshape(emb, 1, T, E, B)
    hidden = relu.(maximum(model.conv1d(emb), dims=2))
    hidden = reshape(hidden, size(hidden,3), B)
    output = model.linear(hidden)
end

(model::CNN)(x,y) = nll(model(x),y)


# In order to make our network working, we need to implement ```Embedding```,
# ```Linear``` and ```Conv``` layers,

mutable struct Embedding; w; end
(layer::Embedding)(x) = layer.w[x, :]
Embedding(vocabsize::Int, embedsize::Int) = Embedding(
    param(vocabsize, embedsize))


mutable struct Linear; w; b; end
(layer::Linear)(x) = layer.w * x .+ layer.b
Linear(inputsize::Int, outputsize::Int) = Linear(
    param(outputsize, inputsize),
    param0(outputsize, 1))


mutable struct Conv; w; b; end
(layer::Conv)(x) = conv4(layer.w, x; stride=1, padding=0) .+ layer.b
Conv(embedsize::Int, nfilters::Int, kernelsize::Int) = Conv(
    param(1, kernelsize, embedsize, nfilters),
    param0(1, 1, nfilters, 1))

# We initialize our model,

EMBEDSIZE = 64
WINSIZE = KERNELSIZE = 3
NFILTERS = 64
model = CNN(
    Embedding(nwords, EMBEDSIZE),
    Conv(EMBEDSIZE, NFILTERS, KERNELSIZE),
    Linear(NFILTERS, ntags))

# We implement a validation procedure which computes accuracy and average loss
# over the entire input data split.

function validate(data)
    loss = correct = 0
    for (x,y) in data
        ŷ = model(x)
        loss += nll(ŷ,y)
        correct += argmax(Array(ŷ))[1] == y[1]
    end
    return loss/length(data), correct/length(data)
end

# Finally, here is the training loop:

function train(nepochs=100)
    for epoch=1:nepochs
        progress!(adam(model, shuffle(trn)))

        trnloss, trnacc = validate(trn)
        @printf("iter %d: trn loss/sent=%.4f, trn acc=%.4f\n",
                epoch, trnloss, trnacc)

        devloss, devacc = validate(dev)
        @printf("iter %d: dev loss/sent=%.4f, dev acc=%.4f\n",
                epoch, devloss, devacc)
    end
end
