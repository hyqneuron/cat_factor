

## Observations

### Softmax Groups
Softmax does in fact have a semantic grouping effect. I wasn't expecting this, but obviously in mlpae4, mlpae7 and
mlpae_bn1, weights under the same softmax tend to be spatially focused in the same region, though this isn't always the
case. Still, there is some kind of weak grouping by softmax.

### Batchnorm turns dead to negative
Softmax alone leaves lots of dead units. Softmax+bn turns dead units into "negative units". All the negative units are
exactly the same.

### Stochastic kills factorization
From X to Xs (turn on stochastic activation), we go from reasonably factorized strokes to whole-letter imprints.
Apparently, this kills factorization.

So, how do we get the factorization to work???


## Logs
Both mlpae and mlpae_bn produced reasonably interesting stroke decompositions. mlpae has more dead units that merely
capture the average input, whereas mlpae_bn turns all those dead units into "negative units".

### mlpae
mlpae produces reasonably factorized strokes. However, the dead units merely capture the average input. It is unclear
whether the dead units contribute to negative probability.

### mlpae_s
Produces whole-letter imprints, and a few dead noisy units. It is completely unclear if any kind of factorization is
going on.

### mlpae_d
mlpae_d learns filters that are a bit "ordered". Though the ordering is not strong and abrupt transitions are quite
common (because we are not doing neighbourhood decay yet). 

It produces reasonably factorized strokes in mlpae_d1 to mlpae_d3

### mlpae_ds
Produces whole-letter imprints, some weird mixed-up transition units and so on.

### mlpae_bn
mlpae_bn produces reasonably factorized strokes. The dead units observed in mlpae are replaced by "negative unit", which
is the negative superposition of the several (3~5) positive modes. The positive modes are spatially close, and all the
negative units are almost identical.

### mlpae_bns
Produces whole-letter imprints, and some dead noisy units. It is completely unclear if any kind of factorization is
going on.

### mlpae_simple
When the number of factors is small (simple1 and simple2), some observable patterns exist. As the number of factors
increase, it becomes essentially high-frequency overfitting. Then dead units start to appear.

