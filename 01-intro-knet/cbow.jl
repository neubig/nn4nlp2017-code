# # Continuous Bag of Words (CBOW) Model

include("bow.jl")

# Instead of mapping words directly to 5 dimensions for 5 classes, we map them to 64
# dimensions to represent features, then go to 5 dimensions after adding up these features.

pred(words) = b_sm .+ W_sm * sum(W_emb[:,words], dims=2)

# Only the prediction function and parameters need to change, the rest of the code is the
# same.

EMB_SIZE = 64
W_emb = param(EMB_SIZE, nwords)
W_sm = param(ntags, EMB_SIZE)
b_sm = param(ntags)

# Bonus question: can this model represent a function that the BOW model cannot represent?

train()
