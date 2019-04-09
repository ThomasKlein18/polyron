# Polyron

Neurons for ANNs with polynomial activation functions.


# To Do 
* try out different degrees and initializations for the coefficients 
* try to learn only one activation function instead of one per neuron (would be a lot faster, question is whether that has any chance of working better than just using relus or tanh everywhere, the whole idea is that neurons can develop different reaction patterns)
* try to find a way of updating the coefficient weights slower than the other weights (that might actually work, no idea how to implement that, though)
  + hack Adam and give them a really low, unchanging learning rate?
* look at the values of the logits in every step
* normalize the logits before feeding them into the activation function (is there a keras layer which I could add to the dense layer that I'm using anyway? Would be neat.)
  + shit, no, there is not... might be worthwhile anyway to implement the Layer Norm paper as a tf.keras layer
