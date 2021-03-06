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

- Datasets can have uniform or non-uniform samples

- A batch is ultimately processed as one tensor (for unsupervised training or
  inference), or two tensors having the same batch size (for supervised training),
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
  samples, we typically drop the remainder, rather than training on a smaller
  batch, because having a smaller batch interferes with BatchNorm and produces
  less accurate gradients.
  
- When training, the ordering of samples must normally be randomized between
  epochs, to minimize the effect of ordering on results.
  
  Sometimes randomization must be specialized: in language models (GPT2),
  training data are sequences of words, order *among* (but not within) the
  sequences is randomized, then uniform samples are collected by slicing the
  concatenation of the the sequences into segments of a constant length, with
  labels collected the same way at an offset of 1 token.

- Processing the batch with the most data first during an epoch is a hack we use
  to avoid memory fragmentation.
    
- When batch shape is a function of the shape of its constituent samples:
  
  - Forming batches from samples of similar shape allows batches to have less
    data on average, which reduces the amount of computation required to
    process a batch.
    
  - To do that, after randomization, we sort each group of K*batchsize samples
    before dividing it into K batches.

## Procedure

Input: base samples, maxBatchSize

1. if training: choose random ordering of base samples
2. Transform base sample set into batch sample set (GPT2)
3. if training: Drop remainder of batch samples
4. if batch samples non-uniform: sort first K*batchSize batch samples by some
  sort key (typically size).
  
    4.1. if inference: K = number of batches  
    4.2. if training: K = some smaller value so as not to destroy all randomness.
  
5. Repeatedly:

    5.1 select first maxBatchSize batch samples
    5.2 choose batch shape
    5.3 create batch tensor by reshaping batch samples and stacking
