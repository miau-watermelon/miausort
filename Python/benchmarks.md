## Key:
| Distribution | Description |
|--------------|-------------|
| gen_random | Even random permutation of integers 0 - n |
| gen_noisy | Random permutation of integers 0 - n in which items are O(sqrt n) from their sorted position |
| gen_few_rand | Random permutation of integers 0 - n in which only n/20 items are swapped |
| gen_scrambled_head | 7/8 sorted, 1/8 random at start |
| gen_scrambled_tail | 7/8 sorted, 1/8 random at end |
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
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.01 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.01 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Grailsort (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.01 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
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
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 |

## List length 100

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.07** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.13 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.01 |
| Grailsort (sqrt n) | 0.18 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.18 | 0.02 | 0.01 |
| Miausort v2 (in-place) | 0.20 | 0.04 | 0.02 |
| Grailsort (in-place) | 0.22 | 0.01 | 0.01 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.04** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.05 | 0.00 | 0.01 |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.10 | 0.00 | 0.01 |
| Miausort v2 (sqrt n) | 0.10 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.11 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.13 | 0.00 | 0.01 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.05** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.05 | 0.00 | 0.01 |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.09 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.09 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 0.10 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.15 | 0.02 | 0.04 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.02** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.01 |
| Miausort v2 (in-place) | 0.07 | 0.00 | 0.01 |
| Miausort v2 (sqrt n) | 0.07 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.13 | 0.00 | 0.01 |
| Grailsort (in-place) | 0.16 | 0.01 | 0.01 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.02** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.05 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.06 | 0.00 | 0.01 |
| Grailsort (sqrt n) | 0.08 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.12 | 0.00 | 0.01 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.07** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.13 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.16 | 0.01 | 0.00 |
| Miausort v2 (sqrt n) | 0.16 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 0.24 | 0.01 | 0.01 |
| Grailsort (in-place) | 0.27 | 0.01 | 0.01 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.02** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.01 |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.09 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.09 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.01 |
| Grailsort (in-place) | 0.18 | 0.01 | 0.04 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v2 (in-place) | **0.00** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 |
| Timsort | 0.00 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.00 |
| Grailsort (in-place) | 0.10 | 0.00 | 0.01 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.01** | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.01 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.02 |
| Grailsort (in-place) | 0.17 | 0.01 | 0.01 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v2 (in-place) | **0.01** | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 |
| Timsort | 0.07 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.30 | 0.01 | 0.01 |
| Grailsort (in-place) | 0.32 | 0.01 | 0.02 |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **1.27** | 0.09 | 0.08 |
| Timsort | 1.33 | 0.04 | 0.03 |
| Miausort v3 (in-place) | 1.94 | 0.15 | 0.11 |
| Grailsort (sqrt n) | 2.43 | 0.08 | 0.06 |
| Miausort v2 (sqrt n) | 2.79 | 0.06 | 0.04 |
| Grailsort (in-place) | 3.05 | 0.05 | 0.04 |
| Miausort v2 (in-place) | 3.20 | 0.08 | 0.06 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.80** | 0.01 | 0.01 |
| Timsort | 0.82 | 0.02 | 0.02 |
| Miausort v3 (in-place) | 1.02 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 1.22 | 0.01 | 0.02 |
| Miausort v2 (sqrt n) | 1.28 | 0.02 | 0.04 |
| Miausort v2 (in-place) | 1.59 | 0.02 | 0.03 |
| Grailsort (in-place) | 2.04 | 0.05 | 0.04 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.83** | 0.04 | 0.06 |
| Timsort | 0.85 | 0.02 | 0.02 |
| Miausort v3 (in-place) | 1.17 | 0.02 | 0.02 |
| Miausort v2 (sqrt n) | 1.19 | 0.02 | 0.02 |
| Miausort v2 (in-place) | 1.53 | 0.04 | 0.05 |
| Grailsort (sqrt n) | 1.63 | 0.07 | 0.05 |
| Grailsort (in-place) | 2.56 | 0.06 | 0.05 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.26** | 0.01 | 0.01 |
| Miausort v3 (sqrt n) | 0.29 | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.49 | 0.01 | 0.02 |
| Miausort v2 (sqrt n) | 0.53 | 0.01 | 0.02 |
| Miausort v2 (in-place) | 0.91 | 0.02 | 0.02 |
| Grailsort (sqrt n) | 1.70 | 0.02 | 0.02 |
| Grailsort (in-place) | 2.51 | 0.03 | 0.03 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.32** | 0.02 | 0.02 |
| Miausort v3 (sqrt n) | 0.34 | 0.02 | 0.01 |
| Miausort v2 (sqrt n) | 0.61 | 0.02 | 0.09 |
| Miausort v3 (in-place) | 0.69 | 0.04 | 0.03 |
| Miausort v2 (in-place) | 0.80 | 0.02 | 0.04 |
| Grailsort (sqrt n) | 1.23 | 0.04 | 0.02 |
| Grailsort (in-place) | 2.00 | 0.06 | 0.09 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **1.26** | 0.03 | 0.03 |
| Timsort | 1.29 | 0.03 | 0.03 |
| Miausort v2 (sqrt n) | 2.51 | 0.15 | 0.08 |
| Miausort v3 (in-place) | 2.93 | 0.08 | 0.08 |
| Grailsort (sqrt n) | 3.45 | 0.03 | 0.03 |
| Miausort v2 (in-place) | 3.60 | 0.08 | 0.06 |
| Grailsort (in-place) | 4.00 | 0.04 | 0.06 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.19** | 0.01 | 0.01 |
| Miausort v3 (sqrt n) | 0.25 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.44 | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.47 | 0.01 | 0.03 |
| Miausort v2 (in-place) | 0.59 | 0.01 | 0.01 |
| Grailsort (sqrt n) | 1.71 | 0.02 | 0.03 |
| Grailsort (in-place) | 2.49 | 0.02 | 0.03 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **0.04** | 0.00 | 0.00 |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.04 | 0.00 | 0.00 |
| Miausort v2 (sqrt n) | 0.04 | 0.00 | 0.00 |
| Timsort | 0.05 | 0.00 | 0.00 |
| Grailsort (sqrt n) | 0.87 | 0.02 | 0.02 |
| Grailsort (in-place) | 1.69 | 0.02 | 0.03 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.08** | 0.00 | 0.00 |
| Miausort v2 (in-place) | 0.14 | 0.01 | 0.01 |
| Miausort v3 (sqrt n) | 0.14 | 0.01 | 0.01 |
| Miausort v3 (in-place) | 0.14 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.14 | 0.01 | 0.08 |
| Grailsort (sqrt n) | 1.67 | 0.05 | 0.04 |
| Grailsort (in-place) | 2.41 | 0.05 | 0.04 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v2 (sqrt n) | **0.15** | 0.01 | 0.00 |
| Miausort v2 (in-place) | 0.15 | 0.01 | 0.00 |
| Miausort v3 (in-place) | 0.15 | 0.00 | 0.00 |
| Miausort v3 (sqrt n) | 0.15 | 0.01 | 0.01 |
| Timsort | 1.15 | 0.03 | 0.02 |
| Grailsort (sqrt n) | 2.88 | 0.03 | 0.04 |
| Grailsort (in-place) | 3.56 | 0.05 | 0.03 |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **15.42** | 0.14 | 0.17 |
| Timsort | 17.99 | 0.10 | 0.08 |
| Miausort v3 (in-place) | 23.55 | 0.22 | 0.16 |
| Grailsort (sqrt n) | 28.19 | 0.10 | 0.09 |
| Miausort v2 (sqrt n) | 30.11 | 0.14 | 0.50 |
| Miausort v2 (in-place) | 37.30 | 0.15 | 0.13 |
| Grailsort (in-place) | 39.86 | 0.23 | 0.52 |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **9.16** | 0.06 | 0.20 |
| Timsort | 10.54 | 0.11 | 0.22 |
| Miausort v3 (in-place) | 12.68 | 0.08 | 0.16 |
| Grailsort (sqrt n) | 14.50 | 0.07 | 0.20 |
| Miausort v2 (sqrt n) | 16.30 | 0.10 | 0.09 |
| Miausort v2 (in-place) | 20.38 | 0.07 | 0.08 |
| Grailsort (in-place) | 26.99 | 0.22 | 0.14 |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **10.75** | 0.06 | 0.05 |
| Miausort v3 (sqrt n) | 10.90 | 0.15 | 0.08 |
| Miausort v2 (sqrt n) | 15.77 | 0.10 | 0.07 |
| Miausort v3 (in-place) | 16.18 | 0.68 | 0.36 |
| Miausort v2 (in-place) | 19.79 | 0.11 | 0.07 |
| Grailsort (sqrt n) | 20.98 | 0.12 | 0.08 |
| Grailsort (in-place) | 34.56 | 1.62 | 0.85 |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **3.27** | 0.03 | 0.03 |
| Timsort | 3.35 | 0.07 | 0.05 |
| Miausort v3 (in-place) | 5.12 | 0.05 | 0.04 |
| Miausort v2 (sqrt n) | 5.65 | 0.06 | 0.06 |
| Miausort v2 (in-place) | 10.32 | 0.06 | 0.06 |
| Grailsort (sqrt n) | 20.36 | 0.42 | 0.25 |
| Grailsort (in-place) | 33.24 | 0.85 | 0.46 |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **3.29** | 0.09 | 0.09 |
| Timsort | 3.39 | 0.07 | 0.16 |
| Miausort v2 (sqrt n) | 6.28 | 0.10 | 0.08 |
| Miausort v3 (in-place) | 6.77 | 0.30 | 0.18 |
| Miausort v2 (in-place) | 9.16 | 0.23 | 0.15 |
| Grailsort (sqrt n) | 15.04 | 0.29 | 0.21 |
| Grailsort (in-place) | 27.82 | 0.27 | 0.35 |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (sqrt n) | **14.39** | 0.08 | 0.05 |
| Timsort | 16.60 | 0.06 | 0.04 |
| Miausort v2 (sqrt n) | 28.44 | 0.98 | 0.51 |
| Miausort v3 (in-place) | 32.10 | 0.08 | 0.19 |
| Miausort v2 (in-place) | 37.96 | 0.13 | 0.11 |
| Grailsort (sqrt n) | 39.98 | 0.15 | 0.21 |
| Grailsort (in-place) | 49.34 | 1.70 | 1.70 |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **1.97** | 0.02 | 0.02 |
| Miausort v3 (sqrt n) | 2.61 | 0.03 | 0.02 |
| Miausort v2 (sqrt n) | 3.69 | 0.05 | 0.05 |
| Miausort v3 (in-place) | 4.06 | 0.05 | 0.04 |
| Miausort v2 (in-place) | 5.13 | 0.08 | 0.06 |
| Grailsort (sqrt n) | 18.98 | 0.12 | 0.22 |
| Grailsort (in-place) | 30.33 | 0.38 | 0.57 |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **0.47** | 0.02 | 0.01 |
| Miausort v3 (sqrt n) | 0.48 | 0.01 | 0.01 |
| Miausort v2 (sqrt n) | 0.48 | 0.02 | 0.01 |
| Miausort v2 (in-place) | 0.48 | 0.01 | 0.02 |
| Timsort | 0.49 | 0.02 | 0.01 |
| Grailsort (sqrt n) | 10.34 | 0.15 | 0.16 |
| Grailsort (in-place) | 23.03 | 1.17 | 0.61 |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Timsort | **0.80** | 0.02 | 0.04 |
| Miausort v2 (in-place) | 1.49 | 0.03 | 0.07 |
| Miausort v2 (sqrt n) | 1.49 | 0.02 | 0.13 |
| Miausort v3 (in-place) | 1.49 | 0.03 | 0.03 |
| Miausort v3 (sqrt n) | 1.50 | 0.03 | 0.05 |
| Grailsort (sqrt n) | 18.56 | 0.65 | 0.42 |
| Grailsort (in-place) | 30.50 | 0.13 | 0.40 |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ |
|-----------|------------|-----|---|
| Miausort v3 (in-place) | **1.54** | 0.03 | 0.03 |
| Miausort v2 (in-place) | 1.54 | 0.02 | 0.02 |
| Miausort v2 (sqrt n) | 1.54 | 0.02 | 0.02 |
| Miausort v3 (sqrt n) | 1.56 | 0.04 | 0.02 |
| Timsort | 13.58 | 0.42 | 0.27 |
| Grailsort (sqrt n) | 31.51 | 0.11 | 0.15 |
| Grailsort (in-place) | 44.35 | 0.14 | 0.42 |
