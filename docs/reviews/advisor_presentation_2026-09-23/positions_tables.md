> Numerical excerpt only; interpretation is in document 37.

# Position x layer probes (pos_v1; C=0.01, 5-fold by origin; AUROC mean over folds)


| condition | position | best layer | best AUROC | L8 | L11 | L14 | L17 | L20 | L22 | L28 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| isolated | last | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| isolated_clean | last | 17 | 0.954 | 0.927 | 0.913 | 0.939 | 0.954 | 0.952 | 0.936 | 0.923 |
| isolated | mean | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| isolated_clean | mean | 14 | 0.961 | 0.950 | 0.945 | 0.961 | 0.958 | 0.954 | 0.948 | 0.922 |
| twins | span_last | 20 | 0.950 | 0.878 | 0.903 | 0.930 | 0.941 | 0.950 | 0.941 | 0.924 |
| twins | span_mean | 17 | 0.972 | 0.951 | 0.951 | 0.959 | 0.972 | 0.961 | 0.950 | 0.915 |
| twins | q_mean | 17 | 0.894 | 0.830 | 0.842 | 0.853 | 0.894 | 0.885 | 0.869 | 0.819 |
| twins | last | 8 | 0.779 | 0.779 | 0.770 | 0.765 | 0.770 | 0.777 | 0.755 | 0.681 |
| edited | span_last | 17 | 0.935 | 0.875 | 0.912 | 0.927 | 0.935 | 0.933 | 0.929 | 0.914 |
| edited | span_mean | 17 | 0.966 | 0.948 | 0.950 | 0.956 | 0.966 | 0.956 | 0.948 | 0.910 |
| edited | q_mean | 17 | 0.884 | 0.818 | 0.827 | 0.861 | 0.884 | 0.866 | 0.847 | 0.796 |
| edited | last | 20 | 0.806 | 0.712 | 0.707 | 0.705 | 0.758 | 0.806 | 0.781 | 0.701 |
| isolated->twins | last->span_last | 20 | 0.618 | 0.527 | 0.537 | 0.470 | 0.480 | 0.618 | 0.558 | 0.553 |
| isolated->twins | mean->span_mean | 14 | 0.620 | 0.582 | 0.586 | 0.620 | 0.618 | 0.582 | 0.581 | 0.607 |
| isolated->edited | last->span_last | 20 | 0.612 | 0.559 | 0.532 | 0.493 | 0.498 | 0.612 | 0.570 | 0.577 |
| isolated->edited | mean->span_mean | 14 | 0.670 | 0.623 | 0.641 | 0.670 | 0.666 | 0.626 | 0.630 | 0.619 |

## Lexical floor on the same pairs (no hidden states)

| condition | readout | AUROC |
|---|---|---:|
| twins | text on span text | 0.875 |
| twins | text on whole question | 0.770 |
| twins | style on span text | 0.821 |
| twins | style on whole question | 0.731 |
| twins | masked on span text | 0.772 |
| twins | masked on whole question | 0.691 |
| edited | text on span text | 0.870 |
| edited | text on whole question | 0.773 |
| edited | style on span text | 0.772 |
| edited | style on whole question | 0.697 |
| edited | masked on span text | 0.749 |
| edited | masked on whole question | 0.692 |
| isolated_clean | text on statement | 0.870 |
| isolated_clean | style on statement | 0.772 |
| isolated_clean | masked on statement | 0.749 |
