# BPNet Locus Diagnostic Summary

This directory contains two sets of diagnostic plots from the BPNet locus
notebook:

- `example1/`: an active locus at which the genomic input is much more active
  than its shuffled references.
- `example2/`: an enhancer locus at which shuffled references are often as
  active as, or more active than, the genomic input.

The analysis compares model folds, dinucleotide-shuffled DeepLIFT reference
banks, attribution methods, and perturbation-based sequence importance.

## Example 1

### Main Findings

- Dinucleotide-shuffled references are not completely neutral. Most have
  moderate predicted activity, and some contain sharp predicted cryptic
  initiators.
- The genomic input is substantially more active than the shuffled references:
  approximately 3,600 predicted counts versus roughly 200-1,000 for most
  references.
- Cryptic reference peaks generally have poor positional agreement across model
  folds. Many therefore appear model-specific rather than robust promoter-like
  signals.
- Profile-head DeepLIFT is sensitive to both reference seed and model fold.
  Seeds 3 and 4 disagree most with the across-seed consensus, while fold 0
  differs notably from folds 1-6.
- Reference activity does not explain profile instability by itself. Seed 0 has
  the highest mean and maximum reference counts but the most stable profile
  attribution. Seed 4 is the least stable without containing the most active
  references.
- Count-head DeepLIFT is highly stable across both seeds and folds. The
  reference-bank sensitivity is therefore primarily a profile-attribution
  problem at this locus.
- DeepLIFT, gradient x input, single-base ISM, and window ISM broadly identify
  the same important sequence regions. This agreement supports the principal
  biological features even where profile DeepLIFT magnitude varies by seed.

### Plot Guide

#### `example1/download.png`: Genomic dinucleotide spectrum

The 16 normalized dinucleotide frequencies in the unmodified 2,114 bp genomic
input. This describes the local sequence composition preserved by
dinucleotide-shuffled references and approximated by Markov replacements.

#### `example1/download (1).png`: Reference activity distributions

Distributions by shuffle seed of profile target score, predicted counts, and
profile Jensen-Shannon divergence from the genomic input. The genomic input is
shown separately by the dashed line.

The references are much less active than the genomic input, although every
seed produces some elevated outliers.

#### `example1/download (2).png`: Count-scaled reference profiles

Plus- and minus-strand heatmaps of:

```text
softmax(profile logits) * exp(log counts)
```

Each row is a shuffled reference averaged across folds. References are grouped
by seed and sorted by their strongest 20 bp signal. Sparse bright regions
identify predicted cryptic initiation sites.

#### `example1/download (3).png`: Centered profile logits

Centered-logit heatmaps using the same ordering as the count-scaled heatmaps.
These separate strong positional preference in the profile head from high
total activity in the count head.

#### `example1/download (4).png`: Total counts versus local activity

Predicted total counts against the maximum 20 bp count-scaled signal. The
genomic input is the black star. The strongest shuffled references are
annotated.

The genomic input is a clear outlier in both total and local activity. One
seed-3 reference has an unusually strong local peak relative to other
references.

#### `example1/download (5).png`: Position-wise reference activity

At each BPNet output position, the plot shows the reference mean, 95th
percentile, maximum, and genomic-input signal for each strand.

References produce isolated peaks throughout the output. The genomic input has
much stronger localized activity, particularly on the minus strand.

#### `example1/download (6).png`: Ranked reference metrics

Ranked distributions of predicted counts, maximum 1 bp, 5 bp, and 20 bp
signals, and profile entropy. The long upper tails show that a small number of
references are considerably more active or concentrated than the rest.

#### `example1/download (7).png`: Fold consensus of reference peaks

The left panel compares peak-position variability with strand agreement. The
right panel shows mean peak position and activity across folds.

Peak positions often vary substantially between folds. Strong agreement is
uncommon, indicating that many cryptic initiators are recognized by only a
subset of models.

#### `example1/download (8).png`: Profile DeepLIFT stability

Reference-seed and model-fold correlation heatmaps for profile attribution.
Seed agreement is moderate, seeds 3 and 4 are least concordant, and fold 0 is
poorly correlated with the other folds.

The position-wise mean and standard-deviation panel in this exported image is
invalid because the notebook previously summed over positions rather than
nucleotide channels. The notebook has been corrected; regenerate this plot for
the proper position-wise profile-attribution variability.

#### `example1/download (9).png`: Count DeepLIFT stability

Reference-seed and model-fold correlations for count attribution. Correlations
are uniformly high, demonstrating that count DeepLIFT is robust to the
reference bank at this locus.

Its position-wise panel was affected by the same plotting bug and should also
be regenerated.

#### `example1/download (10).png`: Profile instability versus reference activity

Profile-attribution disagreement is compared with mean counts, maximum counts,
maximum 20 bp activity, and mean profile entropy for each seed.

There is no simple monotonic relationship. Seed 0 combines the most active
references with the lowest disagreement, while seed 4 has the greatest
disagreement without the strongest reference activity.

#### `example1/download (11).png`: Count instability versus reference activity

The equivalent analysis for count attribution. Disagreement is very small for
all seeds, approximately 0.021-0.026, reinforcing that the count head is
largely insensitive to reference selection.

#### `example1/download (12).png`: Profile attribution methods

Profile-head DeepLIFT/SHAP, gradient x input, and single-base ISM logos.
Although their scales differ, the methods broadly identify the same important
regions, especially toward the left side of the displayed interval.

#### `example1/download (13).png`: Count attribution methods

Count-head DeepLIFT/SHAP, gradient x input, and single-base ISM logos. The
methods agree strongly on prominent sequence features near the right side of
the interval.

#### `example1/download (14).png`: Window ISM

Effects of replacing sliding 10 bp windows with sequences sampled from a
first-order Markov model. Positive values mean the genomic window supports the
selected prediction relative to its replacements.

The profile head depends most strongly on several left-side windows, whereas
the count head depends most strongly on windows near the major right-side
attribution features.

## Example 2: Enhancer Locus

### Main Findings

- The genomic input is not clearly more active than its shuffled references.
  Its predicted count is approximately 600, close to the center of the
  shuffled-reference distribution.
- Many shuffled references exceed the genomic input in predicted counts,
  profile target score, or maximum 20 bp signal. Several references have
  maximum 20 bp activity above 250-300, compared with approximately 140 for the
  genomic input.
- The maximum reference profile exceeds the genomic profile across much of
  both strands. At this locus, shuffled sequences do not behave like a
  low-activity baseline.
- Profile DeepLIFT is unstable across both seeds and folds. Seeds 42 and 69
  disagree most strongly with the seed consensus, and folds 3 and 6 are weakly
  correlated with several other folds.
- Seed 42 combines high reference activity with the greatest profile
  disagreement, providing a direct example of active references affecting the
  attribution result. The relationship is still not universal: seed 4 has the
  strongest maximum 20 bp activity but relatively stable profile attribution.
- Count DeepLIFT remains more stable than profile DeepLIFT, but its
  seed-to-consensus disagreement is larger than in example 1. Seed 4 is the
  clearest count-attribution outlier.
- DeepLIFT, gradient x input, single-base ISM, and window ISM consistently
  identify a broad sequence-sensitive region around the enhancer's strongest
  feature. This supports the biological relevance of that region even though
  the DeepLIFT baseline is problematic.

### Plot Guide

#### `example2/download (15).png`: Genomic dinucleotide spectrum

The enhancer input has especially high `CC` and `GG` frequencies and relatively
low `AT` and `TA` frequencies. Dinucleotide shuffling preserves these aggregate
counts while changing their longer-range arrangement.

#### `example2/download.png`: Reference activity distributions

The genomic input is shown by dashed lines. Unlike example 1, it lies within or
near the shuffled-reference distributions:

- Many references exceed its profile target score.
- The genomic count prediction is near the reference median.
- Reference profiles remain substantially different from the genomic profile,
  with Jensen-Shannon divergences centered near 0.52.

This is the clearest evidence that the shuffled references are not neutral for
this locus and model.

#### `example2/download (1).png`: Count-scaled reference profiles

The reference heatmaps contain numerous localized plus- and minus-strand peaks.
Some rows show coherent patches spanning tens of bases, rather than only weak
isolated noise.

#### `example2/download (2).png`: Centered profile logits

Many references contain strong positive and negative logit structures. These
patterns show that the model assigns pronounced positional preferences even
before multiplication by predicted counts.

#### `example2/download (3).png`: Total counts versus local activity

The genomic input, shown as a black star, is embedded in the reference cloud.
Several references are simultaneously more active overall and more sharply
localized. Examples include references from seeds 4, 6, and 42.

This differs fundamentally from example 1, where the genomic input is a distant
high-activity outlier.

#### `example2/download (4).png`: Position-wise reference activity

The maximum shuffled-reference signal exceeds the genomic signal at many
positions on both strands. The genomic input has a modest minus-strand feature
near output position 660, but shuffled references generate stronger peaks
throughout the output.

The mean and 95th-percentile reference profiles remain lower than most extreme
peaks, indicating that the problem is driven partly by a tail of highly active
references rather than every shuffled sequence.

#### `example2/download (5).png`: Ranked reference metrics

Reference counts extend to approximately 900, maximum 1 bp signals above 200,
and maximum 20 bp signals above 300. The steep upper tails identify a small
group of unusually concentrated and active references.

#### `example2/download (6).png`: Fold consensus of reference peaks

Most references have high peak-position variability across folds. Some strong
references recur at similar positions, but broad disagreement remains the
dominant pattern. Cryptic activity is therefore a mixture of reproducible and
fold-specific effects.

#### `example2/download (7).png`: Profile DeepLIFT stability

Profile attribution has weak-to-moderate seed correlations. Seeds 42 and 69 are
least concordant, while seeds 4 and 6 agree most strongly. Several folds also
show weak pairwise agreement, especially folds 3 and 6.

The exported position-wise panel was generated before the notebook's axis fix
and should be regenerated.

#### `example2/download (8).png`: Count DeepLIFT stability

Count attribution remains strongly correlated across seeds and folds, although
the agreement is visibly weaker than in example 1. This suggests that the
count head is more robust to reference choice but not entirely immune at a
locus where reference activity rivals genomic activity.

Its exported position-wise panel should also be regenerated.

#### `example2/download (9).png`: Profile instability versus reference activity

Seed 42 has both high reference activity and the greatest profile-attribution
disagreement. Seed 47 has relatively low activity and the best agreement. This
supports an activity-instability relationship for some reference banks.

However, seed 4 has the strongest local 20 bp reference activity while
remaining comparatively stable. No single scalar activity metric fully
predicts profile instability.

#### `example2/download (10).png`: Count instability versus reference activity

Seed 4 is the clear count-attribution outlier and also has the strongest local
reference activity. The remaining seeds cluster more closely. This is stronger
evidence of reference activity affecting count attribution than was present in
example 1.

#### `example2/download (11).png`: Profile attribution methods

All three methods identify a broad important region around approximately
`chr1:155,000,100-155,000,250`, including a particularly strong feature near
the right edge of that region. Gradient x input and single-base ISM agree more
closely in magnitude and structure than either agrees with DeepLIFT.

DeepLIFT retains the same broad biological signal but is attenuated and more
baseline-dependent.

#### `example2/download (12).png`: Count attribution methods

DeepLIFT, gradient x input, and single-base ISM agree on the principal
count-supporting feature near `chr1:155,000,220`. The reference-free methods
show stronger effects and additional local structure.

#### `example2/download (13).png`: Window ISM

The strongest profile and count effects occur around the same broad enhancer
region identified by the nucleotide-resolution methods. The large uncertainty
bands show meaningful fold-to-fold variation, particularly around the most
important windows.

No observed-versus-predicted track panel was included in this exported example.

## Cross-Example Interpretation

Together, the examples show two distinct failure regimes:

1. At the active locus in example 1, shuffled references contain cryptic
   initiators but remain far less active than the genomic input. Reference
   activity adds noise but does not explain profile instability by itself.
2. At the enhancer locus in example 2, shuffled-reference activity overlaps or
   exceeds genomic activity. The baseline no longer represents an inactive
   sequence ensemble, and reference activity is more directly associated with
   both profile- and count-attribution disagreement.

Profile DeepLIFT should therefore be summarized across multiple reference
seeds and interpreted alongside gradient x input and ISM. Reporting the
reference activity distribution relative to the genomic input is essential:
the same attribution procedure has a different interpretation when the genomic
input is a strong activity outlier than when it lies inside the reference
distribution.

Count attribution is generally more robust, but example 2 shows that it can
also become reference-sensitive when shuffled-reference activity is comparable
to genomic activity.
