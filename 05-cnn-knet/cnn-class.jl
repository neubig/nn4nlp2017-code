using Knet
using Random, Statistics, Printf


wdict, tdict = Dict(), Dict()
w2i(x) = get!(wdict, x, 1+length(wdict))
t2i(x) = get!(tdict, x, 1+length(tdict))
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

S = w2i("<s>")
UNK = w2i("<unk>")
trn = readdata("../data/classes/train.txt")
w2i(x) = get(wdict, x, UNK)
t2i(x) = tdict[x]
nwords = length(wdict)
ntags = length(tdict)
dev = readdata("../data/classes/test.txt")


mutable struct CNN
    embedding
    conv1d
    linear
end


function (model::CNN)(x)
    emb = model.embedding(x)
    T, E = size(emb); B = 1
    emb = reshape(emb, 1, T, E, B)
    hidden = relu.(maximum(model.conv1d(emb), dims=2))
    hidden = reshape(hidden, size(hidden,3), B)
    output = model.linear(hidden)
end


(model::CNN)(x,y) = nll(model(x),y)


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


EMBEDSIZE = 64
WINSIZE = KERNELSIZE = 3
NFILTERS = 64
model = CNN(
    Embedding(nwords, EMBEDSIZE),
    Conv(EMBEDSIZE, NFILTERS, KERNELSIZE),
    Linear(NFILTERS, ntags))


function validate(data)
    loss = correct = 0
    for (x,y) in data
        ŷ = model(x)
        loss += nll(ŷ,y)
        correct += argmax(Array(ŷ))[1] == y[1]
    end
    return loss/length(data), correct/length(data)
end


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
