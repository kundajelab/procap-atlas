# Model strategies

1) Background choice (GC matched, GC matched + DNAse, whole genome (mask out unmappable regions))
2) Should we switch counts head to Poisson loss?
3) Implement unmappability loss (DONE, but performance looks bad)

Performance of initial draft of BNBPNET looks bad, like much worse than ProCapNet/CLIPNET trained on K562. Need to check if it's:

1) Bug in new training loop or loss function.
2) Don't completely trust peak calls, may need to redo? But Kelly got good results with ENCODE peak set

Attempts:

0) Benchmark ProCapNet
1) Krishna w/ new BNBPNet code
2) Hanuman w/ CLIPNET_pytorch
3) base BPNet
4) Match training loop with CLIPNET_pytorch (TODO)
5) Concat bidirectional + unidirectional peaks