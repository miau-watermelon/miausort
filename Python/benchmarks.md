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

## Algorithms
| Name | Runtime (best) | Runtime (worst) | Space complexity | Stable? | Adaptivity |
|------|----------------|-----------------|------------------|---------|------------|
| Grailsort | O(n log n) | O(n log n) | O(1), O(sqrt n) | Yes | None |
| Timsort | O(n) | O(n log n) | O(n) | Yes | Run scans, galloping |
| Introsort | O(n log n) | O(n log n) | O(log n) | No | None | 
| Miausort v2 | O(n) | O(n log n) | O(1), O(sqrt n), O(n) | Yes | Run scans, galloping |
| Miausort v3 | O(n) | O(n log n) | O(1), O(sqrt n), O(n) | Yes | Run scans |

# Benchmark (50 runs)

## List length 10

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 |0.00 | 0.00 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (sqrt n) | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Introsort | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Introsort | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 |0.00 | 0.00 | 100.0% |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Introsort | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 |0.00 | 0.00 | 100.0% |

## List length 100

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.03** |0.00 | 0.00 | 100.0% |
| Timsort | 0.07 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.11 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.13 |0.00 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.13 |0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.13 |0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.17 |0.01 | 0.04 | 100.0% |
| Miausort v2 (in-place) | 0.17 |0.01 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 0.18 |0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.20 |0.02 | 0.05 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.02** |0.00 | 0.00 | 100.0% |
| Timsort | 0.05 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.06 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.07 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.10 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.10 |0.01 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 0.10 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.13 |0.01 | 0.01 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.01** |0.00 | 0.00 | 100.0% |
| Timsort | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.05 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.05 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.05 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.10 |0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.10 |0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.14 |0.01 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.15 |0.01 | 0.01 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.02** |0.00 | 0.00 | 100.0% |
| Introsort | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.07 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.07 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.12 |0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.16 |0.01 | 0.01 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.02** |0.00 | 0.00 | 100.0% |
| Introsort | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.06 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.06 |0.01 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.08 |0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.11 |0.00 | 0.00 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.03** |0.00 | 0.00 | 100.0% |
| Timsort | 0.07 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.11 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.13 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.16 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.16 |0.00 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.16 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.18 |0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.25 |0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.27 |0.01 | 0.01 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.02** |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.06 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 |0.00 | 0.00 | 100.0% |
| Introsort | 0.07 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.08 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.09 |0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.14 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.17 |0.01 | 0.01 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.00** |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 |0.00 | 0.00 | 100.0% |
| Timsort | 0.00 |0.00 | 0.00 | 100.0% |
| Introsort | 0.01 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.07 |0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.11 |0.02 | 0.03 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.01** |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.01 |0.00 | 0.01 | 100.0% |
| Introsort | 0.02 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 |0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.16 |0.01 | 0.01 | 100.0% |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v2 (in-place) | **0.01** |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 |0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.01 |0.00 | 0.00 | 100.0% |
| Introsort | 0.02 |0.00 | 0.00 | 100.0% |
| Timsort | 0.07 |0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.29 |0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.32 |0.01 | 0.07 | 100.0% |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.54** |0.02 | 0.02 | 100.0% |
| Miausort v3 (n/2) | 1.01 |0.02 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.27 |0.03 | 0.03 | 100.0% |
| Timsort | 1.29 |0.05 | 0.07 | 100.0% |
| Miausort v2 (n/2) | 1.71 |0.05 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 1.91 |0.06 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 2.24 |0.04 | 0.03 | 100.0% |
| Miausort v2 (sqrt n) | 2.70 |0.10 | 0.06 | 100.0% |
| Grailsort (in-place) | 2.94 |0.06 | 0.05 | 100.0% |
| Miausort v2 (in-place) | 3.06 |0.08 | 0.09 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.49** |0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.69 |0.01 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 0.79 |0.02 | 0.03 | 100.0% |
| Timsort | 0.80 |0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 1.02 |0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.20 |0.02 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 1.24 |0.03 | 0.03 | 100.0% |
| Miausort v2 (sqrt n) | 1.34 |0.05 | 0.07 | 100.0% |
| Miausort v2 (in-place) | 1.68 |0.02 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.03 |0.04 | 0.06 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.25** |0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.62 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.71 |0.02 | 0.01 | 100.0% |
| Timsort | 0.77 |0.02 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 0.79 |0.02 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.06 |0.01 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 1.08 |0.02 | 0.03 | 100.0% |
| Miausort v2 (in-place) | 1.29 |0.02 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 1.58 |0.03 | 0.04 | 100.0% |
| Grailsort (in-place) | 2.38 |0.05 | 0.04 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.22** |0.01 | 0.01 | 100.0% |
| Timsort | 0.26 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.28 |0.01 | 0.01 | 100.0% |
| Miausort v2 (n/2) | 0.32 |0.01 | 0.01 | 100.0% |
| Introsort | 0.45 |0.02 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.48 |0.02 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.51 |0.01 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.88 |0.05 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 1.70 |0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.50 |0.05 | 0.04 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.25** |0.01 | 0.01 | 100.0% |
| Timsort | 0.27 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.32 |0.01 | 0.01 | 100.0% |
| Miausort v2 (n/2) | 0.37 |0.01 | 0.02 | 100.0% |
| Introsort | 0.42 |0.01 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 0.58 |0.02 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.66 |0.03 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 0.76 |0.03 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.13 |0.02 | 0.02 | 100.0% |
| Grailsort (in-place) | 1.90 |0.04 | 0.03 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.46** |0.01 | 0.02 | 100.0% |
| Miausort v3 (n/2) | 0.94 |0.03 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 1.19 |0.04 | 0.03 | 100.0% |
| Timsort | 1.24 |0.03 | 0.03 | 100.0% |
| Miausort v2 (n/2) | 1.62 |0.03 | 0.03 | 100.0% |
| Miausort v2 (sqrt n) | 2.48 |0.05 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 2.94 |0.11 | 0.06 | 100.0% |
| Grailsort (sqrt n) | 3.27 |0.06 | 0.04 | 100.0% |
| Miausort v2 (in-place) | 3.58 |0.12 | 0.10 | 100.0% |
| Grailsort (in-place) | 3.76 |0.08 | 0.08 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.18** |0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.22 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.23 |0.01 | 0.01 | 100.0% |
| Miausort v2 (n/2) | 0.24 |0.01 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 0.41 |0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.44 |0.02 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.55 |0.02 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.59 |0.05 | 0.03 | 100.0% |
| Introsort | 1.79 |0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.29 |0.05 | 0.05 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.04** |0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.04 |0.00 | 0.01 | 100.0% |
| Miausort v2 (n/2) | 0.04 |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 |0.00 | 0.00 | 100.0% |
| Timsort | 0.05 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.05 |0.00 | 0.00 | 100.0% |
| Introsort | 0.22 |0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.90 |0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 1.71 |0.04 | 0.03 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.07** |0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.14 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.14 |0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.14 |0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 |0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.14 |0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.14 |0.00 | 0.01 | 100.0% |
| Introsort | 0.25 |0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.64 |0.06 | 0.06 | 100.0% |
| Grailsort (in-place) | 2.26 |0.05 | 0.06 | 100.0% |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (in-place) | **0.15** |0.01 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.15 |0.01 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.15 |0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.15 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.15 |0.01 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.15 |0.01 | 0.01 | 100.0% |
| Introsort | 0.25 |0.01 | 0.01 | 100.0% |
| Timsort | 1.16 |0.03 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 2.92 |0.05 | 0.04 | 100.0% |
| Grailsort (in-place) | 3.61 |0.06 | 0.26 | 100.0% |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **8.56** |0.08 | 0.09 | 100.0% |
| Miausort v3 (n/2) | 13.59 |0.30 | 0.22 | 100.0% |
| Miausort v3 (sqrt n) | 14.81 |0.50 | 0.70 | 100.0% |
| Timsort | 17.57 |0.15 | 0.18 | 100.0% |
| Miausort v2 (n/2) | 21.52 |0.11 | 0.24 | 100.0% |
| Miausort v3 (in-place) | 23.60 |0.89 | 0.67 | 100.0% |
| Grailsort (sqrt n) | 26.99 |0.19 | 0.20 | 100.0% |
| Miausort v2 (sqrt n) | 29.31 |0.53 | 0.36 | 100.0% |
| Miausort v2 (in-place) | 35.58 |0.14 | 0.31 | 100.0% |
| Grailsort (in-place) | 37.72 |0.80 | 0.57 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **5.91** |0.05 | 0.04 | 100.0% |
| Miausort v3 (n/2) | 8.55 |0.18 | 0.14 | 100.0% |
| Miausort v3 (sqrt n) | 8.68 |0.27 | 0.16 | 100.0% |
| Timsort | 10.50 |0.11 | 0.16 | 100.0% |
| Miausort v3 (in-place) | 12.76 |0.10 | 0.09 | 100.0% |
| Grailsort (sqrt n) | 14.30 |0.17 | 0.25 | 100.0% |
| Miausort v2 (n/2) | 14.68 |0.30 | 0.19 | 100.0% |
| Miausort v2 (sqrt n) | 16.66 |0.41 | 0.23 | 100.0% |
| Miausort v2 (in-place) | 20.49 |0.13 | 0.12 | 100.0% |
| Grailsort (in-place) | 26.66 |0.18 | 0.13 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **3.82** |0.03 | 0.03 | 100.0% |
| Miausort v3 (n/2) | 9.54 |0.06 | 0.05 | 100.0% |
| Timsort | 10.36 |0.08 | 0.06 | 100.0% |
| Miausort v3 (sqrt n) | 10.60 |0.08 | 0.07 | 100.0% |
| Miausort v2 (n/2) | 12.62 |0.06 | 0.04 | 100.0% |
| Miausort v2 (sqrt n) | 15.93 |0.12 | 0.09 | 100.0% |
| Miausort v3 (in-place) | 16.33 |0.08 | 0.08 | 100.0% |
| Miausort v2 (in-place) | 19.24 |0.71 | 0.37 | 100.0% |
| Grailsort (sqrt n) | 21.13 |0.19 | 0.19 | 100.0% |
| Grailsort (in-place) | 33.77 |0.24 | 0.39 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **2.79** |0.03 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 3.18 |0.03 | 0.04 | 100.0% |
| Timsort | 3.21 |0.03 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 3.88 |0.04 | 0.04 | 100.0% |
| Miausort v3 (in-place) | 5.00 |0.03 | 0.19 | 100.0% |
| Miausort v2 (sqrt n) | 5.63 |0.05 | 0.05 | 100.0% |
| Introsort | 6.52 |0.05 | 0.04 | 100.0% |
| Miausort v2 (in-place) | 9.66 |0.05 | 0.14 | 100.0% |
| Grailsort (sqrt n) | 20.12 |0.16 | 0.18 | 100.0% |
| Grailsort (in-place) | 31.99 |0.39 | 0.32 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **2.93** |0.06 | 0.07 | 100.0% |
| Miausort v3 (sqrt n) | 3.21 |0.08 | 0.08 | 100.0% |
| Timsort | 3.34 |0.13 | 0.11 | 100.0% |
| Miausort v2 (n/2) | 3.98 |0.11 | 0.09 | 100.0% |
| Miausort v2 (sqrt n) | 6.05 |0.11 | 0.12 | 100.0% |
| Introsort | 6.45 |0.15 | 0.15 | 100.0% |
| Miausort v3 (in-place) | 6.57 |0.12 | 0.14 | 100.0% |
| Miausort v2 (in-place) | 8.63 |0.19 | 0.22 | 100.0% |
| Grailsort (sqrt n) | 14.14 |0.19 | 0.36 | 100.0% |
| Grailsort (in-place) | 25.63 |0.46 | 0.46 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **6.00** |0.08 | 0.15 | 100.0% |
| Miausort v3 (n/2) | 12.57 |0.38 | 0.30 | 100.0% |
| Miausort v3 (sqrt n) | 14.11 |0.44 | 0.26 | 100.0% |
| Timsort | 16.99 |0.91 | 0.61 | 100.0% |
| Miausort v2 (n/2) | 20.37 |0.65 | 0.49 | 100.0% |
| Miausort v2 (sqrt n) | 27.96 |0.80 | 0.60 | 100.0% |
| Miausort v3 (in-place) | 31.04 |1.16 | 0.71 | 100.0% |
| Miausort v2 (in-place) | 37.05 |0.90 | 1.11 | 100.0% |
| Grailsort (sqrt n) | 38.10 |0.42 | 0.33 | 100.0% |
| Grailsort (in-place) | 47.41 |0.33 | 0.31 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **2.01** |0.05 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 2.63 |0.09 | 0.06 | 100.0% |
| Miausort v2 (n/2) | 2.80 |0.09 | 0.08 | 100.0% |
| Miausort v3 (n/2) | 2.95 |0.09 | 0.10 | 100.0% |
| Miausort v2 (sqrt n) | 3.81 |0.09 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 4.12 |0.09 | 0.07 | 100.0% |
| Miausort v2 (in-place) | 5.29 |0.14 | 0.13 | 100.0% |
| Grailsort (sqrt n) | 19.21 |0.37 | 0.34 | 100.0% |
| Grailsort (in-place) | 29.76 |0.47 | 0.41 | 100.0% |
| Introsort | 33.52 |0.57 | 0.66 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (in-place) | **0.46** |0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.47 |0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.47 |0.01 | 0.01 | 100.0% |
| Miausort v2 (n/2) | 0.47 |0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.47 |0.01 | 0.01 | 100.0% |
| Timsort | 0.49 |0.02 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.51 |0.01 | 0.01 | 100.0% |
| Introsort | 3.39 |0.14 | 0.09 | 100.0% |
| Grailsort (sqrt n) | 10.41 |0.16 | 0.22 | 100.0% |
| Grailsort (in-place) | 22.56 |0.19 | 0.30 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.79** |0.02 | 0.02 | 100.0% |
| Miausort v3 (n/2) | 1.52 |0.04 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 1.53 |0.04 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.54 |0.03 | 0.03 | 100.0% |
| Miausort v2 (n/2) | 1.54 |0.02 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 1.56 |0.03 | 0.18 | 100.0% |
| Miausort v2 (in-place) | 1.56 |0.03 | 0.09 | 100.0% |
| Introsort | 3.72 |0.06 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 18.88 |0.18 | 0.32 | 100.0% |
| Grailsort (in-place) | 28.35 |0.31 | 0.34 | 100.0% |

### gen_reversed_steps

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (in-place) | **1.49** |0.03 | 0.02 | 100.0% |
| Miausort v3 (n/2) | 1.50 |0.03 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.50 |0.02 | 0.05 | 100.0% |
| Miausort v2 (n/2) | 1.51 |0.03 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 1.55 |0.04 | 0.04 | 100.0% |
| Miausort v2 (in-place) | 1.59 |0.02 | 0.02 | 100.0% |
| Introsort | 3.68 |0.06 | 0.04 | 100.0% |
| Timsort | 12.40 |0.10 | 0.10 | 100.0% |
| Grailsort (sqrt n) | 31.92 |0.91 | 0.60 | 100.0% |
| Grailsort (in-place) | 41.60 |0.25 | 0.27 | 100.0% |
