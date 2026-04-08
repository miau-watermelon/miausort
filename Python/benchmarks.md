## Key:
| Distribution | Description |
|--------------|-------------|
| gen_random | Even random permutation of integers 0 - n |
| gen_noisy | Random permutation of integers 0 - n in which items are O(sqrt n) from their sorted position |
| gen_few_rand | Random permutation of integers 0 - n in which only n/20 items are swapped |
| gen_scrambled_head | 7/8 sorted, 1/8 random at start |
| gen_scrambled_tail | 7/8 sorted, 1/8 random at end |
| gen_sawtooth | Four sorted runs |
| gen_sqrtn_unique | Even random permutation of integers 0 - sqrt n with each value repeated sqrt n times |
| gen_pipe_organ | Half sorted from 0 - n/2, half reversed from n/2 - 0 | 
| gen_sorted | Sorted array of integers 0 - n |
| gen_reversed | Reverse sorted array of integers 0 - n |
| gen_reversed_steps | Reverse sorted array of integers 0 - n/2 with each value repeated twice |

# Benchmark (50 runs)

## List length 10

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 |
| Timsort | **0.00** | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 |

## List length 100

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.07** | 0.01 | 0.00 |
| Miausort v3 (in-place) | 0.13 | 0.01 | 0.00 |
| Miausort v3 (sqrt n) | 0.13 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 0.16 | 0.01 | 0.01 |
| Miausort v2 (in-place) | 0.17 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.17 | 0.01 | 0.01 |
| Grailsort (in-place) | 0.20 | 0.01 | 0.01 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.04** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.05 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.09 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.09 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.09 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.13 | 0.01 | 0.01 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.05** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.05 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.11 | 0.00 | 0.01 |
| Grailsort (sqrt n) | 0.11 | 0.00 | 0.01 |
| Miausort v2 (in-place) | 0.13 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.15 | 0.01 | 0.00 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.02** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.07 | 0.00 | 0.01 |
| Miausort v2 (sqrt n) | 0.07 | 0.00 | 0.01 |
| Grailsort (sqrt n) | 0.13 | 0.01 | 0.01 |
| Grailsort (in-place) | 0.17 | 0.01 | 0.01 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.02** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.01 |
| Miausort v2 (sqrt n) | 0.06 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.06 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.11 | 0.01 | 0.02 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.07** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.13 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.16 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.17 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 0.23 | 0.01 | 0.02 |
| Grailsort (in-place) | 0.26 | 0.01 | 0.02 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.02** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.06 | 0.01 | 0.01 |
| Miausort v2 (in-place) | 0.08 | 0.00 | 0.01 |
| Miausort v2 (sqrt n) | 0.09 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.01 |
| Grailsort (in-place) | 0.17 | 0.01 | 0.01 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v2 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.06 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.10 | 0.00 | 0.02 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.01** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.01 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.13 | 0.00 | 0.01 |
| Grailsort (in-place) | 0.16 | 0.01 | 0.04 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.01** | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.01 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.01 |
| Timsort | 0.06 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.29 | 0.03 | 0.02 |
| Grailsort (in-place) | 0.32 | 0.01 | 0.02 |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **1.29** | 0.03 | 0.03 |
| Timsort | 1.31 | 0.03 | 0.03 |
| Miausort v3 (in-place) | 1.92 | 0.10 | 0.08 |
| Grailsort (sqrt n) | 2.23 | 0.05 | 0.07 |
| Miausort v2 (sqrt n) | 2.59 | 0.08 | 0.09 |
| Grailsort (in-place) | 2.90 | 0.06 | 0.06 |
| Miausort v2 (in-place) | 3.04 | 0.10 | 0.07 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.78** | 0.01 | 0.01 |
| Timsort | 0.79 | 0.02 | 0.01 |
| Miausort v3 (in-place) | 1.01 | 0.02 | 0.04 |
| Grailsort (sqrt n) | 1.22 | 0.06 | 0.05 |
| Miausort v2 (sqrt n) | 1.30 | 0.02 | 0.01 |
| Miausort v2 (in-place) | 1.63 | 0.03 | 0.02 |
| Grailsort (in-place) | 2.04 | 0.04 | 0.05 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.76** | 0.02 | 0.01 |
| Timsort | 0.79 | 0.02 | 0.02 |
| Miausort v3 (in-place) | 1.13 | 0.02 | 0.02 |
| Miausort v2 (sqrt n) | 1.13 | 0.02 | 0.02 |
| Miausort v2 (in-place) | 1.38 | 0.02 | 0.02 |
| Grailsort (sqrt n) | 1.67 | 0.03 | 0.04 |
| Grailsort (in-place) | 2.54 | 0.03 | 0.05 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.27** | 0.01 | 0.01 |
| Miausort v3 (sqrt n) | 0.30 | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.51 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.53 | 0.01 | 0.01 |
| Miausort v2 (in-place) | 0.93 | 0.02 | 0.02 |
| Grailsort (sqrt n) | 1.79 | 0.03 | 0.04 |
| Grailsort (in-place) | 2.62 | 0.04 | 0.07 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.29** | 0.01 | 0.01 |
| Miausort v3 (sqrt n) | 0.34 | 0.02 | 0.01 |
| Miausort v2 (sqrt n) | 0.61 | 0.03 | 0.07 |
| Miausort v3 (in-place) | 0.70 | 0.03 | 0.02 |
| Miausort v2 (in-place) | 0.82 | 0.04 | 0.04 |
| Grailsort (sqrt n) | 1.19 | 0.06 | 0.05 |
| Grailsort (in-place) | 2.00 | 0.04 | 0.05 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **1.24** | 0.03 | 0.02 |
| Timsort | 1.27 | 0.03 | 0.02 |
| Miausort v2 (sqrt n) | 2.44 | 0.05 | 0.04 |
| Miausort v3 (in-place) | 2.89 | 0.15 | 0.08 |
| Grailsort (sqrt n) | 3.30 | 0.08 | 0.06 |
| Miausort v2 (in-place) | 3.54 | 0.09 | 0.06 |
| Grailsort (in-place) | 3.72 | 0.08 | 0.06 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.18** | 0.01 | 0.01 |
| Miausort v3 (sqrt n) | 0.24 | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.45 | 0.01 | 0.02 |
| Miausort v2 (sqrt n) | 0.49 | 0.07 | 0.04 |
| Miausort v2 (in-place) | 0.68 | 0.04 | 0.02 |
| Grailsort (sqrt n) | 1.62 | 0.04 | 0.04 |
| Grailsort (in-place) | 2.29 | 0.06 | 0.04 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.04** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.05 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.05 | 0.00 | 0.00 |
| Timsort | 0.05 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.91 | 0.01 | 0.02 |
| Grailsort (in-place) | 1.74 | 0.03 | 0.02 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.07** | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.15 | 0.00 | 0.02 |
| Miausort v3 (sqrt n) | 0.15 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.15 | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.15 | 0.01 | 0.00 |
| Grailsort (sqrt n) | 1.69 | 0.07 | 0.04 |
| Grailsort (in-place) | 2.40 | 0.03 | 0.05 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.15** | 0.01 | 0.00 |
| Miausort v3 (in-place) | 0.15 | 0.01 | 0.00 |
| Miausort v2 (sqrt n) | 0.15 | 0.00 | 0.01 |
| Miausort v2 (in-place) | 0.15 | 0.01 | 0.01 |
| Timsort | 1.15 | 0.03 | 0.02 |
| Grailsort (sqrt n) | 2.93 | 0.05 | 0.04 |
| Grailsort (in-place) | 3.65 | 0.08 | 0.09 |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **14.69** | 0.15 | 0.13 |
| Timsort | 17.13 | 0.17 | 0.31 |
| Miausort v3 (in-place) | 23.62 | 0.23 | 0.23 |
| Grailsort (sqrt n) | 27.72 | 0.16 | 0.12 |
| Miausort v2 (sqrt n) | 29.69 | 1.30 | 2.19 |
| Miausort v2 (in-place) | 35.95 | 0.90 | 0.61 |
| Grailsort (in-place) | 38.22 | 1.67 | 0.87 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **8.77** | 0.06 | 0.04 |
| Timsort | 10.52 | 0.04 | 0.05 |
| Miausort v3 (in-place) | 12.73 | 0.12 | 0.07 |
| Grailsort (sqrt n) | 14.69 | 0.51 | 0.23 |
| Miausort v2 (sqrt n) | 15.72 | 0.15 | 0.15 |
| Miausort v2 (in-place) | 19.70 | 0.52 | 0.97 |
| Grailsort (in-place) | 26.72 | 0.16 | 0.12 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **10.11** | 0.34 | 0.17 |
| Miausort v3 (sqrt n) | 10.32 | 0.05 | 0.04 |
| Miausort v2 (sqrt n) | 15.27 | 0.13 | 0.11 |
| Miausort v3 (in-place) | 16.17 | 0.11 | 0.21 |
| Miausort v2 (in-place) | 19.42 | 0.62 | 0.36 |
| Grailsort (sqrt n) | 22.39 | 0.51 | 0.33 |
| Grailsort (in-place) | 33.33 | 0.19 | 0.26 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **3.24** | 0.03 | 0.03 |
| Miausort v3 (sqrt n) | 3.28 | 0.03 | 0.02 |
| Miausort v3 (in-place) | 5.16 | 0.04 | 0.04 |
| Miausort v2 (sqrt n) | 5.65 | 0.05 | 0.03 |
| Miausort v2 (in-place) | 9.82 | 0.10 | 0.21 |
| Grailsort (sqrt n) | 19.61 | 0.51 | 0.29 |
| Grailsort (in-place) | 33.11 | 0.29 | 0.59 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **3.18** | 0.04 | 0.03 |
| Timsort | 3.34 | 0.11 | 0.08 |
| Miausort v2 (sqrt n) | 5.99 | 0.09 | 0.09 |
| Miausort v3 (in-place) | 6.50 | 0.13 | 0.21 |
| Miausort v2 (in-place) | 8.79 | 0.25 | 0.18 |
| Grailsort (sqrt n) | 14.42 | 0.13 | 0.09 |
| Grailsort (in-place) | 26.59 | 0.18 | 0.14 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **14.60** | 0.09 | 0.06 |
| Timsort | 16.66 | 0.38 | 0.21 |
| Miausort v2 (sqrt n) | 28.17 | 0.11 | 0.08 |
| Miausort v3 (in-place) | 32.31 | 0.10 | 0.12 |
| Miausort v2 (in-place) | 37.32 | 1.54 | 0.81 |
| Grailsort (sqrt n) | 38.39 | 0.26 | 0.28 |
| Grailsort (in-place) | 46.70 | 1.60 | 0.79 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **1.92** | 0.03 | 0.02 |
| Miausort v3 (sqrt n) | 2.55 | 0.03 | 0.03 |
| Miausort v2 (sqrt n) | 3.63 | 0.03 | 0.18 |
| Miausort v3 (in-place) | 3.99 | 0.04 | 0.03 |
| Miausort v2 (in-place) | 5.00 | 0.04 | 0.03 |
| Grailsort (sqrt n) | 18.66 | 0.12 | 0.09 |
| Grailsort (in-place) | 29.57 | 0.21 | 0.27 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.46** | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.46 | 0.01 | 0.01 |
| Timsort | 0.47 | 0.01 | 0.02 |
| Miausort v2 (sqrt n) | 0.48 | 0.01 | 0.01 |
| Miausort v2 (in-place) | 0.49 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 10.61 | 0.12 | 0.12 |
| Grailsort (in-place) | 23.04 | 0.15 | 0.23 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.79** | 0.02 | 0.01 |
| Miausort v2 (sqrt n) | 1.53 | 0.03 | 0.02 |
| Miausort v2 (in-place) | 1.54 | 0.03 | 0.02 |
| Miausort v3 (sqrt n) | 1.56 | 0.02 | 0.02 |
| Miausort v3 (in-place) | 1.67 | 0.14 | 0.07 |
| Grailsort (sqrt n) | 18.78 | 0.11 | 0.14 |
| Grailsort (in-place) | 28.30 | 0.17 | 0.14 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **1.54** | 0.02 | 0.02 |
| Miausort v2 (sqrt n) | 1.55 | 0.03 | 0.02 |
| Miausort v3 (sqrt n) | 1.55 | 0.03 | 0.03 |
| Miausort v2 (in-place) | 1.55 | 0.03 | 0.03 |
| Timsort | 12.59 | 0.09 | 0.05 |
| Grailsort (sqrt n) | 31.24 | 0.42 | 0.28 |
| Grailsort (in-place) | 42.47 | 1.13 | 0.59 |
