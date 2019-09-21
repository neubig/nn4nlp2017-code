# # Deep Continuous Bag of Words (DeepCBOW) Model

include("bow.jl")

# This time after mapping the words to embeddings and adding up these embeddings we pass
# them through a couple linear + tanh layers.

function pred(words)
    h = sum(W_emb[:,words], dims=2)
    for (w,b) in zip(W_h, b_h)
        h = tanh.(w * h .+ b)
    end
    return W_sm * h .+ b_sm
end

# Again, we only need to change the prediction function and the parameters:

EMB_SIZE = 64
HID_SIZE = 64
HID_LAY = 2
W_emb = param(EMB_SIZE, nwords)
W_h = [ param(HID_SIZE, lay == 1 ? EMB_SIZE : HID_SIZE) for lay in 1:HID_LAY ]
b_h = [ param(HID_SIZE) for lay in 1:HID_LAY ]
W_sm = param(ntags, HID_SIZE)
b_sm = param(ntags)

# Bonus question: can this model represent a function that the CBOW model cannot represent?

train()
