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
| gen_reversed_duplicates | Reverse sorted array of integers 0 - n/2 with each value repeated twice |

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
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | **0.00** | 0.00 | 0.00 | 100.0% |
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v2 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v2 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |

## List length 100

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.11 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.13 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.13 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.17 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.17 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.18 | 0.00 | 0.02 | 100.0% |
| Grailsort (in-place) | 0.20 | 0.01 | 0.01 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.09 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.10 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.10 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.12 | 0.00 | 0.00 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.04 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.09 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.09 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.11 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.14 | 0.00 | 0.01 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.18 | 0.02 | 0.02 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.05 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.11 | 0.00 | 0.01 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.10 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.12 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.12 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.12 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.16 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.16 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.24 | 0.01 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.27 | 0.01 | 0.03 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.08 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.09 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.00 | 0.01 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.10 | 0.00 | 0.00 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.02 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.13 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.16 | 0.01 | 0.04 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v2 (in-place) | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.02 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.28 | 0.00 | 0.02 | 100.0% |
| Grailsort (in-place) | 0.31 | 0.01 | 0.01 | 100.0% |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.70** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.97 | 0.03 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.27 | 0.03 | 0.06 | 100.0% |
| Timsort | 1.29 | 0.05 | 0.03 | 100.0% |
| Miausort v2 (n/2) | 1.70 | 0.03 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.92 | 0.04 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 2.21 | 0.03 | 0.04 | 100.0% |
| Miausort v2 (sqrt n) | 2.63 | 0.05 | 0.04 | 100.0% |
| Grailsort (in-place) | 2.90 | 0.04 | 0.03 | 100.0% |
| Miausort v2 (in-place) | 3.02 | 0.07 | 0.05 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.46** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (n/2) | 0.68 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.77 | 0.01 | 0.01 | 100.0% |
| Timsort | 0.81 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 1.00 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.20 | 0.01 | 0.05 | 100.0% |
| Miausort v2 (n/2) | 1.21 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 1.30 | 0.01 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 1.64 | 0.01 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.01 | 0.04 | 0.02 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.25** | 0.01 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.73 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.75 | 0.01 | 0.02 | 100.0% |
| Timsort | 0.79 | 0.01 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 0.82 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 1.13 | 0.02 | 0.22 | 100.0% |
| Miausort v2 (sqrt n) | 1.16 | 0.02 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 1.38 | 0.03 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 1.69 | 0.04 | 0.11 | 100.0% |
| Grailsort (in-place) | 2.56 | 0.04 | 0.03 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.22** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.26 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.28 | 0.01 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.32 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.47 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.49 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.53 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.89 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.70 | 0.03 | 0.05 | 100.0% |
| Grailsort (in-place) | 2.50 | 0.03 | 0.03 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **0.25** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.28 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.32 | 0.02 | 0.01 | 100.0% |
| Miausort v2 (n/2) | 0.37 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.45 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.58 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.67 | 0.02 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 0.79 | 0.03 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.12 | 0.02 | 0.01 | 100.0% |
| Grailsort (in-place) | 1.89 | 0.07 | 0.05 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **0.45** | 0.01 | 0.02 | 100.0% |
| Miausort v3 (n/2) | 0.93 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 1.18 | 0.03 | 0.02 | 100.0% |
| Timsort | 1.20 | 0.03 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 1.57 | 0.03 | 0.04 | 100.0% |
| Miausort v2 (sqrt n) | 2.33 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 2.62 | 0.08 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 3.19 | 0.05 | 0.03 | 100.0% |
| Miausort v2 (in-place) | 3.34 | 0.12 | 0.07 | 100.0% |
| Grailsort (in-place) | 3.74 | 0.05 | 0.05 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.18** | 0.01 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 0.24 | 0.01 | 0.03 | 100.0% |
| Miausort v2 (n/2) | 0.25 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (n/2) | 0.26 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.40 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.45 | 0.01 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 0.54 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.56 | 0.02 | 0.01 | 100.0% |
| Introsort | 1.81 | 0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.27 | 0.03 | 0.18 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (sqrt n) | **0.04** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.05 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.22 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.89 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 1.68 | 0.03 | 0.03 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.07** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.16 | 0.01 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.16 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (n/2) | 0.16 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.16 | 0.01 | 0.00 | 100.0% |
| Introsort | 0.24 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 1.77 | 0.03 | 0.04 | 100.0% |
| Grailsort (in-place) | 2.29 | 0.04 | 0.08 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v2 (n/2) | **0.14** | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (n/2) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.16 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.24 | 0.01 | 0.01 | 100.0% |
| Timsort | 1.09 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 2.82 | 0.05 | 0.05 | 100.0% |
| Grailsort (in-place) | 3.46 | 0.04 | 0.14 | 100.0% |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **7.68** | 0.12 | 0.08 | 100.0% |
| Miausort v3 (n/2) | 12.76 | 0.45 | 0.26 | 100.0% |
| Miausort v3 (sqrt n) | 14.92 | 0.10 | 0.06 | 100.0% |
| Timsort | 17.00 | 0.09 | 0.08 | 100.0% |
| Miausort v2 (n/2) | 21.22 | 0.08 | 0.11 | 100.0% |
| Miausort v3 (in-place) | 23.96 | 0.07 | 0.10 | 100.0% |
| Grailsort (sqrt n) | 26.74 | 0.16 | 0.13 | 100.0% |
| Miausort v2 (sqrt n) | 28.72 | 0.66 | 0.36 | 100.0% |
| Miausort v2 (in-place) | 36.30 | 0.43 | 0.52 | 100.0% |
| Grailsort (in-place) | 37.94 | 0.24 | 0.33 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **5.93** | 0.06 | 0.04 | 100.0% |
| Miausort v3 (n/2) | 8.52 | 0.08 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 8.63 | 0.08 | 0.05 | 100.0% |
| Timsort | 10.41 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 12.58 | 0.23 | 0.12 | 100.0% |
| Miausort v2 (n/2) | 14.22 | 0.29 | 0.22 | 100.0% |
| Grailsort (sqrt n) | 14.38 | 0.11 | 0.12 | 100.0% |
| Miausort v2 (sqrt n) | 16.30 | 0.08 | 0.06 | 100.0% |
| Miausort v2 (in-place) | 20.27 | 0.11 | 0.09 | 100.0% |
| Grailsort (in-place) | 26.42 | 0.55 | 0.31 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **3.74** | 0.06 | 0.04 | 100.0% |
| Miausort v3 (n/2) | 9.22 | 0.04 | 0.04 | 100.0% |
| Timsort | 10.08 | 0.07 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 10.16 | 0.14 | 0.09 | 100.0% |
| Miausort v2 (n/2) | 11.80 | 0.06 | 0.04 | 100.0% |
| Miausort v2 (sqrt n) | 15.11 | 0.06 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 15.76 | 0.06 | 0.04 | 100.0% |
| Miausort v2 (in-place) | 19.00 | 0.82 | 0.40 | 100.0% |
| Grailsort (sqrt n) | 20.90 | 0.15 | 0.10 | 100.0% |
| Grailsort (in-place) | 33.14 | 0.16 | 0.11 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **2.90** | 0.04 | 0.03 | 100.0% |
| Timsort | 3.26 | 0.11 | 0.14 | 100.0% |
| Miausort v3 (sqrt n) | 3.29 | 0.04 | 0.04 | 100.0% |
| Miausort v2 (n/2) | 4.02 | 0.06 | 0.06 | 100.0% |
| Miausort v3 (in-place) | 5.23 | 0.06 | 0.05 | 100.0% |
| Miausort v2 (sqrt n) | 5.73 | 0.05 | 0.05 | 100.0% |
| Introsort | 6.61 | 0.08 | 0.06 | 100.0% |
| Miausort v2 (in-place) | 9.58 | 0.07 | 0.08 | 100.0% |
| Grailsort (sqrt n) | 19.66 | 0.82 | 0.41 | 100.0% |
| Grailsort (in-place) | 32.35 | 0.87 | 0.44 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (n/2) | **2.93** | 0.05 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 3.22 | 0.06 | 0.06 | 100.0% |
| Timsort | 3.31 | 0.06 | 0.08 | 100.0% |
| Miausort v2 (n/2) | 3.92 | 0.12 | 0.11 | 100.0% |
| Miausort v2 (sqrt n) | 5.96 | 0.08 | 0.06 | 100.0% |
| Miausort v3 (in-place) | 6.50 | 0.13 | 0.10 | 100.0% |
| Introsort | 6.94 | 0.09 | 0.08 | 100.0% |
| Miausort v2 (in-place) | 8.76 | 0.12 | 0.18 | 100.0% |
| Grailsort (sqrt n) | 14.14 | 0.12 | 0.27 | 100.0% |
| Grailsort (in-place) | 26.57 | 0.22 | 0.36 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Introsort | **6.15** | 0.14 | 0.11 | 100.0% |
| Miausort v3 (n/2) | 12.50 | 0.20 | 0.14 | 100.0% |
| Miausort v3 (sqrt n) | 14.04 | 0.20 | 0.41 | 100.0% |
| Timsort | 16.24 | 0.49 | 0.27 | 100.0% |
| Miausort v2 (n/2) | 20.18 | 0.26 | 0.38 | 100.0% |
| Miausort v2 (sqrt n) | 27.25 | 0.33 | 0.30 | 100.0% |
| Miausort v3 (in-place) | 29.18 | 0.65 | 0.40 | 100.0% |
| Miausort v2 (in-place) | 36.80 | 0.98 | 0.57 | 100.0% |
| Grailsort (sqrt n) | 38.29 | 0.88 | 0.56 | 100.0% |
| Grailsort (in-place) | 46.84 | 2.14 | 1.09 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **2.01** | 0.04 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 2.66 | 0.06 | 0.04 | 100.0% |
| Miausort v2 (n/2) | 2.80 | 0.07 | 0.04 | 100.0% |
| Miausort v3 (n/2) | 3.00 | 0.11 | 0.07 | 100.0% |
| Miausort v2 (sqrt n) | 3.77 | 0.07 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 4.18 | 0.12 | 0.09 | 100.0% |
| Miausort v2 (in-place) | 5.18 | 0.11 | 0.07 | 100.0% |
| Grailsort (sqrt n) | 19.21 | 0.17 | 0.16 | 100.0% |
| Grailsort (in-place) | 30.39 | 0.93 | 0.53 | 100.0% |
| Introsort | 34.57 | 0.46 | 0.49 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v3 (in-place) | **0.48** | 0.02 | 0.05 | 100.0% |
| Miausort v3 (n/2) | 0.48 | 0.02 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 0.48 | 0.01 | 0.03 | 100.0% |
| Miausort v2 (sqrt n) | 0.49 | 0.02 | 0.04 | 100.0% |
| Miausort v2 (n/2) | 0.49 | 0.01 | 0.04 | 100.0% |
| Timsort | 0.50 | 0.01 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 0.50 | 0.01 | 0.04 | 100.0% |
| Introsort | 3.48 | 0.09 | 0.06 | 100.0% |
| Grailsort (sqrt n) | 10.89 | 0.43 | 0.24 | 100.0% |
| Grailsort (in-place) | 23.32 | 0.45 | 0.41 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Timsort | **0.77** | 0.02 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 1.48 | 0.03 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 1.48 | 0.04 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.49 | 0.03 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 1.49 | 0.03 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.50 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (n/2) | 1.56 | 0.09 | 0.05 | 100.0% |
| Introsort | 3.61 | 0.05 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 18.19 | 0.11 | 0.11 | 100.0% |
| Grailsort (in-place) | 28.31 | 0.15 | 0.10 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | σ | Correct % |
|-----------|-------------|-----|---|-----------|
| Miausort v2 (sqrt n) | **1.54** | 0.03 | 0.02 | 100.0% |
| Miausort v2 (n/2) | 1.56 | 0.02 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.57 | 0.03 | 0.13 | 100.0% |
| Miausort v2 (in-place) | 1.57 | 0.03 | 0.07 | 100.0% |
| Miausort v3 (n/2) | 1.58 | 0.02 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.59 | 0.06 | 0.07 | 100.0% |
| Introsort | 3.79 | 0.04 | 0.22 | 100.0% |
| Timsort | 13.02 | 0.15 | 0.09 | 100.0% |
| Grailsort (sqrt n) | 31.72 | 0.16 | 0.20 | 100.0% |
| Grailsort (in-place) | 42.60 | 0.17 | 0.33 | 100.0% |
