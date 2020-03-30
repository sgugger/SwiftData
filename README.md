# Batching

Deep learning uses a technique called “batching” to better utilize hardware
resources by processing many data samples as one large tensor:

- Initiating data transfer to an accelerator can be a costly synchronization
  point; fewer synchronizations lead to higher performance.
  
- Accelerators typically produce performance wins by repeating the same
  calculation over and over, in parallel, without interruption.  Computations on
  large tensors are easily broken down into similar pieces that can be handled
  at once.
  
- When training, updating weights based on gradients calculated from every
  sample becomes another bottleneck that prevents large quantities of
  data from being handled at once.  By computing the average gradient over
  larger batches and updating weights once for the whole batch, we can reduce
  the number and frequency of weight updates.

## Terminology

Per https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/:

* Batch Gradient Descent. Batch Size = Size of Training Set
* Stochastic Gradient Descent. Batch Size = 1
* Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set

## The Problem Space

- datasets can have uniform or non-uniform samples

- A batch is ultimately processed as one tensor (for unsupervised training or
  inference), or two tensors having the same shape (for supervised training),
  having rank one higher than that of the samples, with the added (“sample”)
  dimension indexing data corresponding to each sample.  The shape of the batch
  in the other dimensions is usually, but not always, the union of the shapes of
  the samples.
  
- When a slice of the batch across the sample dimension doesn't match the shape
  of the corresponding sample, the data in that slice must be chosen.  Often
  that's done by padding with zeros, but it could involve a nontrivial
  transformation such as image resizing.  In general this is an arbitrary
  reshaping function that maps the sample to the batch slice.
  
- When training, if the batch size doesn't evenly divide the total number of
  samples, we typically (always?) drop the remainder, rather than padding the
  last batch.
  
- When training, the ordering of samples must normally be randomized between
  epochs, to minimize the effect of ordering on results.
  
  Sometimes randomization must be specialized: in language models (GPT2), order
  is randomized keeping phrases intact, then uniform samples are collected by
  slicing the concatenation of the phrases into segments of a constant length,
  with labels collected the same way at an offset of 1 token.

- Processing the batch with the most data first during an epoch is a hack we use
  to avoid memory fragmentation.
    
- When batch shape is a function of the shape of its constituent samples:
  
  - Forming batches from samples of similar shape allows batches to have less
    data on average, which reduces the amount of computation required to
    process a batch.
    
  - To do that, we sort each group of K*batchsize samples before dividing it
    into K batches.
