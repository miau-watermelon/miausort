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

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.04 | 0.00 | 0.01 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |

## List length 100

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.06 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.13 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.16 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.19 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.25 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.38 | 0.01 | 0.02 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.09 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.12 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.23 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.33 | 0.00 | 0.01 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.04 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.11 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.15 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.21 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.28 | 0.00 | 0.01 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.03 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.13 | 0.01 | 0.03 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.18 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.24 | 0.00 | 0.00 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.02 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.08 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.12 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.19 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.26 | 0.00 | 0.01 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.03 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.01 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.10 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.10 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.13 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.19 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.27 | 0.00 | 0.02 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.06 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.12 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.24 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.26 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.27 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.43 | 0.00 | 0.01 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.02 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.17 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.24 | 0.00 | 0.00 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.04 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.02 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.19 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.20 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.23 | 0.00 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.29 | 0.00 | 0.04 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.04** | 0.00 | 0.01 | 100.0% |
| Powersort | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.09 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.09 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.10 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.17 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.20 | 0.00 | 0.05 | 100.0% |
| Grailsort (in-place) | 0.21 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.29 | 0.00 | 0.01 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.01 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.10 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.10 | 0.00 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 0.26 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.27 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.29 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.49 | 0.00 | 0.01 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.08 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.10 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.13 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.19 | 0.00 | 0.01 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.01** | 0.00 | 0.00 | 100.0% |
| Powersort | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.13 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.16 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.17 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.23 | 0.00 | 0.01 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | **0.01** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.02 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.01 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.21 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.28 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.31 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.34 | 0.00 | 0.00 | 100.0% |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.56** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 1.27 | 0.09 | 0.06 | 100.0% |
| Timsort | 1.28 | 0.04 | 0.04 | 100.0% |
| Powersort | 1.33 | 0.03 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 1.86 | 0.07 | 0.07 | 100.0% |
| Grailsort (sqrt n) | 2.23 | 0.07 | 0.18 | 100.0% |
| Wiki Sort (sqrt n) | 2.60 | 0.07 | 0.06 | 100.0% |
| Grailsort (in-place) | 2.87 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 4.54 | 0.19 | 0.11 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.46** | 0.00 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 0.75 | 0.01 | 0.01 | 100.0% |
| Timsort | 0.77 | 0.02 | 0.01 | 100.0% |
| Powersort | 0.82 | 0.01 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 0.95 | 0.02 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 1.16 | 0.03 | 0.14 | 100.0% |
| Grailsort (in-place) | 1.92 | 0.04 | 0.05 | 100.0% |
| Wiki Sort (sqrt n) | 2.11 | 0.02 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 3.59 | 0.03 | 0.03 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.24** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.71 | 0.01 | 0.03 | 100.0% |
| Timsort | 0.74 | 0.01 | 0.15 | 100.0% |
| Powersort | 0.78 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 1.11 | 0.01 | 0.08 | 100.0% |
| Grailsort (sqrt n) | 1.67 | 0.02 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 2.06 | 0.02 | 0.04 | 100.0% |
| Grailsort (in-place) | 2.44 | 0.04 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 3.19 | 0.03 | 0.07 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.25** | 0.01 | 0.02 | 100.0% |
| Powersort | 0.26 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.28 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.44 | 0.00 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.48 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.26 | 0.05 | 0.06 | 100.0% |
| Grailsort (sqrt n) | 1.69 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.70 | 0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.45 | 0.05 | 0.03 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.28** | 0.01 | 0.01 | 100.0% |
| Powersort | 0.28 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.33 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.47 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.68 | 0.02 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.21 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.28 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 1.79 | 0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 1.99 | 0.04 | 0.04 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.29** | 0.00 | 0.01 | 100.0% |
| Timsort | 0.30 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.40 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.71 | 0.02 | 0.01 | 100.0% |
| Introsort | 1.23 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.23 | 0.03 | 0.06 | 100.0% |
| Grailsort (sqrt n) | 1.66 | 0.21 | 0.20 | 100.0% |
| Wiki Sort (in-place) | 1.72 | 0.03 | 0.09 | 100.0% |
| Grailsort (in-place) | 2.44 | 0.05 | 0.03 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.47** | 0.01 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.15 | 0.03 | 0.03 | 100.0% |
| Timsort | 1.16 | 0.02 | 0.02 | 100.0% |
| Powersort | 1.22 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 2.54 | 0.08 | 0.04 | 100.0% |
| Miausort v3 (in-place) | 2.65 | 0.07 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 3.21 | 0.05 | 0.03 | 100.0% |
| Grailsort (in-place) | 3.71 | 0.08 | 0.05 | 100.0% |
| Wiki Sort (in-place) | 4.61 | 0.21 | 0.31 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.18** | 0.01 | 0.00 | 100.0% |
| Powersort | 0.18 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.23 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.45 | 0.01 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 1.28 | 0.02 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 1.58 | 0.02 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.60 | 0.02 | 0.03 | 100.0% |
| Introsort | 1.79 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.25 | 0.04 | 0.04 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.47** | 0.01 | 0.02 | 100.0% |
| Powersort | 0.50 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.51 | 0.01 | 0.02 | 100.0% |
| Introsort | 0.61 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.85 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 2.18 | 0.03 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 2.43 | 0.05 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 3.20 | 0.04 | 0.03 | 100.0% |
| Grailsort (in-place) | 3.20 | 0.05 | 0.04 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.59** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.60 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.62 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.95 | 0.02 | 0.14 | 100.0% |
| Wiki Sort (sqrt n) | 1.71 | 0.03 | 0.02 | 100.0% |
| Introsort | 1.99 | 0.03 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 2.07 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 2.73 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.77 | 0.07 | 0.15 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.41** | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 1.07 | 0.04 | 0.04 | 100.0% |
| Timsort | 1.14 | 0.02 | 0.02 | 100.0% |
| Powersort | 1.17 | 0.03 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 2.22 | 0.07 | 0.04 | 100.0% |
| Wiki Sort (sqrt n) | 2.54 | 0.07 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 3.50 | 0.05 | 0.08 | 100.0% |
| Grailsort (in-place) | 4.03 | 0.06 | 0.11 | 100.0% |
| Wiki Sort (in-place) | 4.69 | 0.16 | 0.09 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.04** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.22 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.81 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.90 | 0.06 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 0.99 | 0.01 | 0.03 | 100.0% |
| Grailsort (in-place) | 1.67 | 0.08 | 0.11 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.07** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.08 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.24 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.46 | 0.02 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 1.65 | 0.02 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.67 | 0.02 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.29 | 0.04 | 0.04 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.14** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.25 | 0.00 | 0.00 | 100.0% |
| Timsort | 1.11 | 0.02 | 0.13 | 100.0% |
| Powersort | 1.17 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.88 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 2.84 | 0.08 | 0.10 | 100.0% |
| Wiki Sort (in-place) | 3.12 | 0.03 | 0.04 | 100.0% |
| Grailsort (in-place) | 3.42 | 0.06 | 0.08 | 100.0% |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **9.37** | 0.32 | 0.49 | 100.0% |
| Miausort v3 (sqrt n) | 15.87 | 0.19 | 0.34 | 100.0% |
| Timsort | 19.02 | 0.46 | 0.52 | 100.0% |
| Powersort | 20.42 | 0.37 | 0.48 | 100.0% |
| Miausort v3 (in-place) | 25.34 | 0.68 | 0.59 | 100.0% |
| Wiki Sort (sqrt n) | 26.02 | 0.86 | 0.68 | 100.0% |
| Grailsort (sqrt n) | 28.95 | 0.33 | 0.53 | 100.0% |
| Grailsort (in-place) | 41.46 | 1.07 | 0.89 | 100.0% |
| Wiki Sort (in-place) | 62.37 | 1.45 | 2.07 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.46** | 0.09 | 0.11 | 100.0% |
| Miausort v3 (sqrt n) | 8.83 | 0.08 | 0.09 | 100.0% |
| Timsort | 10.47 | 0.09 | 0.09 | 100.0% |
| Powersort | 12.00 | 0.26 | 0.70 | 100.0% |
| Miausort v3 (in-place) | 13.14 | 0.14 | 0.16 | 100.0% |
| Grailsort (sqrt n) | 15.00 | 0.18 | 0.14 | 100.0% |
| Wiki Sort (sqrt n) | 20.11 | 0.29 | 0.49 | 100.0% |
| Grailsort (in-place) | 27.58 | 0.24 | 0.18 | 100.0% |
| Wiki Sort (in-place) | 46.39 | 0.43 | 0.67 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **4.40** | 0.07 | 0.08 | 100.0% |
| Timsort | 10.42 | 0.18 | 0.38 | 100.0% |
| Miausort v3 (sqrt n) | 10.53 | 0.18 | 0.12 | 100.0% |
| Powersort | 11.55 | 0.23 | 0.14 | 100.0% |
| Miausort v3 (in-place) | 17.03 | 0.35 | 0.28 | 100.0% |
| Wiki Sort (sqrt n) | 20.54 | 0.51 | 0.60 | 100.0% |
| Grailsort (sqrt n) | 22.01 | 0.41 | 0.30 | 100.0% |
| Grailsort (in-place) | 35.15 | 0.57 | 0.39 | 100.0% |
| Wiki Sort (in-place) | 40.09 | 0.36 | 0.71 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **3.32** | 0.05 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 3.35 | 0.05 | 0.04 | 100.0% |
| Powersort | 3.52 | 0.07 | 0.10 | 100.0% |
| Miausort v3 (in-place) | 5.26 | 0.06 | 0.31 | 100.0% |
| Introsort | 6.84 | 0.17 | 0.14 | 100.0% |
| Wiki Sort (sqrt n) | 10.86 | 0.10 | 0.22 | 100.0% |
| Wiki Sort (in-place) | 17.17 | 0.50 | 0.30 | 100.0% |
| Grailsort (sqrt n) | 20.90 | 0.19 | 0.16 | 100.0% |
| Grailsort (in-place) | 33.37 | 0.47 | 0.65 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **3.18** | 0.05 | 0.05 | 100.0% |
| Timsort | 3.34 | 0.06 | 0.05 | 100.0% |
| Powersort | 3.44 | 0.08 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 6.49 | 0.07 | 0.08 | 100.0% |
| Introsort | 7.01 | 0.06 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 10.24 | 0.20 | 0.13 | 100.0% |
| Grailsort (sqrt n) | 14.45 | 0.49 | 0.27 | 100.0% |
| Wiki Sort (in-place) | 15.91 | 0.32 | 0.56 | 100.0% |
| Grailsort (in-place) | 27.37 | 0.52 | 0.36 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **3.33** | 0.07 | 0.09 | 100.0% |
| Powersort | 3.34 | 0.08 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 3.75 | 0.09 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 6.03 | 0.11 | 0.09 | 100.0% |
| Wiki Sort (sqrt n) | 10.53 | 0.65 | 0.34 | 100.0% |
| Wiki Sort (in-place) | 15.28 | 0.68 | 0.34 | 100.0% |
| Introsort | 18.96 | 0.20 | 0.16 | 100.0% |
| Grailsort (sqrt n) | 20.13 | 0.31 | 0.38 | 100.0% |
| Grailsort (in-place) | 32.51 | 0.97 | 0.61 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.81** | 0.13 | 0.20 | 100.0% |
| Miausort v3 (sqrt n) | 15.79 | 0.21 | 0.37 | 100.0% |
| Timsort | 17.58 | 0.76 | 0.46 | 100.0% |
| Powersort | 19.01 | 0.38 | 0.55 | 100.0% |
| Wiki Sort (sqrt n) | 25.92 | 1.46 | 0.87 | 100.0% |
| Miausort v3 (in-place) | 32.09 | 1.48 | 1.03 | 100.0% |
| Grailsort (sqrt n) | 40.96 | 0.87 | 0.55 | 100.0% |
| Grailsort (in-place) | 49.96 | 0.80 | 0.99 | 100.0% |
| Wiki Sort (in-place) | 64.19 | 2.73 | 2.04 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **2.18** | 0.12 | 0.13 | 100.0% |
| Timsort | 2.20 | 0.10 | 0.07 | 100.0% |
| Miausort v3 (sqrt n) | 2.86 | 0.13 | 0.09 | 100.0% |
| Miausort v3 (in-place) | 4.46 | 0.12 | 0.10 | 100.0% |
| Wiki Sort (sqrt n) | 12.11 | 0.24 | 0.25 | 100.0% |
| Wiki Sort (in-place) | 16.20 | 0.21 | 0.31 | 100.0% |
| Grailsort (sqrt n) | 20.48 | 0.27 | 0.85 | 100.0% |
| Grailsort (in-place) | 32.08 | 0.80 | 0.69 | 100.0% |
| Introsort | 37.95 | 1.02 | 1.02 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **4.49** | 0.38 | 0.21 | 100.0% |
| Powersort | 4.59 | 0.43 | 0.22 | 100.0% |
| Introsort | 7.43 | 0.41 | 0.27 | 100.0% |
| Miausort v3 (sqrt n) | 8.02 | 0.50 | 0.26 | 100.0% |
| Miausort v3 (in-place) | 12.69 | 0.60 | 0.31 | 100.0% |
| Wiki Sort (sqrt n) | 19.89 | 0.89 | 0.47 | 100.0% |
| Wiki Sort (in-place) | 30.60 | 2.30 | 1.36 | 100.0% |
| Grailsort (sqrt n) | 31.88 | 0.42 | 0.73 | 100.0% |
| Grailsort (in-place) | 45.99 | 2.10 | 1.24 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **7.04** | 0.09 | 0.09 | 100.0% |
| Timsort | 7.23 | 0.10 | 0.15 | 100.0% |
| Powersort | 7.87 | 0.06 | 0.06 | 100.0% |
| Miausort v3 (in-place) | 10.90 | 0.21 | 0.18 | 100.0% |
| Wiki Sort (sqrt n) | 15.72 | 0.14 | 0.15 | 100.0% |
| Grailsort (sqrt n) | 24.53 | 0.25 | 0.17 | 100.0% |
| Wiki Sort (in-place) | 32.06 | 0.25 | 0.19 | 100.0% |
| Introsort | 32.48 | 0.65 | 0.42 | 100.0% |
| Grailsort (in-place) | 37.62 | 0.36 | 0.48 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.04** | 0.06 | 0.08 | 100.0% |
| Miausort v3 (sqrt n) | 12.51 | 0.25 | 0.26 | 100.0% |
| Timsort | 13.75 | 0.14 | 0.11 | 100.0% |
| Powersort | 15.25 | 0.24 | 0.24 | 100.0% |
| Wiki Sort (sqrt n) | 23.95 | 0.45 | 0.51 | 100.0% |
| Miausort v3 (in-place) | 25.87 | 0.14 | 0.17 | 100.0% |
| Grailsort (sqrt n) | 41.32 | 0.70 | 0.57 | 100.0% |
| Grailsort (in-place) | 50.57 | 0.75 | 1.03 | 100.0% |
| Wiki Sort (in-place) | 55.70 | 1.31 | 0.79 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.48** | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.49 | 0.01 | 0.02 | 100.0% |
| Timsort | 0.49 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.50 | 0.01 | 0.02 | 100.0% |
| Introsort | 3.65 | 0.05 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 6.26 | 0.10 | 0.15 | 100.0% |
| Wiki Sort (in-place) | 7.21 | 0.12 | 0.37 | 100.0% |
| Grailsort (sqrt n) | 10.93 | 0.42 | 0.49 | 100.0% |
| Grailsort (in-place) | 23.82 | 0.49 | 0.33 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.82** | 0.02 | 0.01 | 100.0% |
| Timsort | 0.84 | 0.02 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 1.52 | 0.06 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 1.56 | 0.05 | 0.04 | 100.0% |
| Introsort | 3.96 | 0.09 | 0.07 | 100.0% |
| Wiki Sort (sqrt n) | 14.17 | 0.49 | 0.28 | 100.0% |
| Wiki Sort (in-place) | 17.32 | 0.41 | 0.47 | 100.0% |
| Grailsort (sqrt n) | 19.52 | 0.52 | 0.37 | 100.0% |
| Grailsort (in-place) | 30.35 | 0.76 | 0.56 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **1.65** | 0.04 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 1.65 | 0.05 | 0.05 | 100.0% |
| Introsort | 4.20 | 0.12 | 0.09 | 100.0% |
| Timsort | 14.04 | 0.14 | 0.10 | 100.0% |
| Powersort | 15.37 | 0.46 | 0.27 | 100.0% |
| Wiki Sort (sqrt n) | 17.37 | 0.58 | 0.84 | 100.0% |
| Grailsort (sqrt n) | 33.02 | 0.91 | 0.63 | 100.0% |
| Wiki Sort (in-place) | 35.99 | 0.72 | 0.62 | 100.0% |
| Grailsort (in-place) | 44.96 | 0.84 | 1.08 | 100.0% |

## List length 100000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **125.19** | 13.07 | 9.17 | 100.0% |
| Miausort v3 (sqrt n) | 201.88 | 9.08 | 6.18 | 100.0% |
| Timsort | 236.65 | 8.06 | 6.24 | 100.0% |
| Powersort | 249.36 | 11.23 | 10.34 | 100.0% |
| Wiki Sort (sqrt n) | 308.68 | 15.42 | 9.63 | 100.0% |
| Miausort v3 (in-place) | 328.78 | 16.62 | 9.70 | 100.0% |
| Grailsort (sqrt n) | 349.47 | 13.20 | 7.70 | 100.0% |
| Grailsort (in-place) | 502.57 | 26.87 | 17.65 | 100.0% |
| Wiki Sort (in-place) | 721.90 | 24.56 | 13.84 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **81.52** | 1.72 | 1.89 | 100.0% |
| Miausort v3 (sqrt n) | 108.80 | 3.71 | 3.05 | 100.0% |
| Timsort | 132.66 | 2.86 | 1.93 | 100.0% |
| Powersort | 145.74 | 2.56 | 1.71 | 100.0% |
| Miausort v3 (in-place) | 170.22 | 4.53 | 5.67 | 100.0% |
| Grailsort (sqrt n) | 186.19 | 5.70 | 5.41 | 100.0% |
| Wiki Sort (sqrt n) | 225.25 | 5.34 | 3.69 | 100.0% |
| Grailsort (in-place) | 354.77 | 13.20 | 51.97 | 100.0% |
| Wiki Sort (in-place) | 541.05 | 11.86 | 7.20 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **58.00** | 0.91 | 0.82 | 100.0% |
| Timsort | 137.23 | 3.92 | 2.64 | 100.0% |
| Powersort | 144.21 | 6.78 | 4.19 | 100.0% |
| Miausort v3 (sqrt n) | 144.45 | 3.65 | 5.34 | 100.0% |
| Miausort v3 (in-place) | 234.71 | 7.15 | 4.16 | 100.0% |
| Wiki Sort (sqrt n) | 252.28 | 9.40 | 5.57 | 100.0% |
| Grailsort (sqrt n) | 278.38 | 7.13 | 4.96 | 100.0% |
| Grailsort (in-place) | 446.54 | 13.20 | 8.07 | 100.0% |
| Wiki Sort (in-place) | 500.64 | 10.99 | 11.49 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **38.77** | 1.07 | 0.89 | 100.0% |
| Timsort | 40.41 | 1.44 | 0.81 | 100.0% |
| Powersort | 41.89 | 1.38 | 1.00 | 100.0% |
| Miausort v3 (in-place) | 60.32 | 1.59 | 1.28 | 100.0% |
| Introsort | 105.55 | 2.80 | 2.77 | 100.0% |
| Wiki Sort (sqrt n) | 108.95 | 2.86 | 2.41 | 100.0% |
| Wiki Sort (in-place) | 178.15 | 6.59 | 4.59 | 100.0% |
| Grailsort (sqrt n) | 243.11 | 5.82 | 4.17 | 100.0% |
| Grailsort (in-place) | 415.13 | 8.01 | 6.75 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **41.06** | 1.11 | 2.18 | 100.0% |
| Powersort | 41.48 | 1.21 | 1.20 | 100.0% |
| Miausort v3 (sqrt n) | 41.81 | 2.34 | 1.49 | 100.0% |
| Miausort v3 (in-place) | 81.17 | 2.26 | 1.58 | 100.0% |
| Introsort | 90.03 | 1.47 | 1.19 | 100.0% |
| Wiki Sort (sqrt n) | 100.82 | 3.75 | 2.45 | 100.0% |
| Wiki Sort (in-place) | 168.16 | 5.47 | 3.72 | 100.0% |
| Grailsort (sqrt n) | 184.45 | 5.01 | 3.96 | 100.0% |
| Grailsort (in-place) | 359.47 | 8.53 | 6.42 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **33.75** | 1.17 | 1.02 | 100.0% |
| Powersort | 33.88 | 1.11 | 1.62 | 100.0% |
| Miausort v3 (sqrt n) | 38.61 | 0.87 | 0.61 | 100.0% |
| Miausort v3 (in-place) | 63.56 | 1.53 | 0.88 | 100.0% |
| Wiki Sort (sqrt n) | 93.80 | 2.70 | 1.95 | 100.0% |
| Wiki Sort (in-place) | 140.19 | 3.19 | 2.32 | 100.0% |
| Grailsort (sqrt n) | 236.78 | 5.15 | 4.10 | 100.0% |
| Introsort | 304.55 | 6.27 | 4.50 | 100.0% |
| Grailsort (in-place) | 410.92 | 10.08 | 7.10 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **105.54** | 3.31 | 2.70 | 100.0% |
| Miausort v3 (sqrt n) | 204.90 | 12.10 | 7.65 | 100.0% |
| Timsort | 235.13 | 10.53 | 7.61 | 100.0% |
| Powersort | 252.45 | 10.92 | 11.02 | 100.0% |
| Wiki Sort (sqrt n) | 314.64 | 13.32 | 9.20 | 100.0% |
| Miausort v3 (in-place) | 409.73 | 19.21 | 17.60 | 100.0% |
| Grailsort (sqrt n) | 509.68 | 15.60 | 10.01 | 100.0% |
| Grailsort (in-place) | 658.67 | 15.37 | 12.15 | 100.0% |
| Wiki Sort (in-place) | 798.98 | 24.49 | 21.67 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **20.66** | 0.39 | 0.39 | 100.0% |
| Timsort | 20.81 | 0.68 | 0.65 | 100.0% |
| Miausort v3 (sqrt n) | 28.15 | 1.21 | 0.98 | 100.0% |
| Miausort v3 (in-place) | 42.82 | 1.36 | 1.58 | 100.0% |
| Wiki Sort (sqrt n) | 118.83 | 3.00 | 1.96 | 100.0% |
| Wiki Sort (in-place) | 153.54 | 3.25 | 2.29 | 100.0% |
| Grailsort (sqrt n) | 234.80 | 4.80 | 4.44 | 100.0% |
| Grailsort (in-place) | 400.85 | 10.00 | 6.63 | 100.0% |
| Introsort | 498.13 | 9.57 | 6.29 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **51.81** | 1.76 | 1.11 | 100.0% |
| Timsort | 53.03 | 1.57 | 1.21 | 100.0% |
| Miausort v3 (sqrt n) | 96.60 | 3.20 | 2.20 | 100.0% |
| Introsort | 99.46 | 2.06 | 1.80 | 100.0% |
| Miausort v3 (in-place) | 147.84 | 3.28 | 2.77 | 100.0% |
| Wiki Sort (sqrt n) | 204.59 | 4.68 | 3.70 | 100.0% |
| Wiki Sort (in-place) | 310.50 | 8.93 | 6.69 | 100.0% |
| Grailsort (sqrt n) | 385.81 | 7.36 | 5.33 | 100.0% |
| Grailsort (in-place) | 570.84 | 12.81 | 8.42 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **81.50** | 2.30 | 2.10 | 100.0% |
| Miausort v3 (sqrt n) | 83.97 | 2.49 | 1.61 | 100.0% |
| Powersort | 88.32 | 3.26 | 2.11 | 100.0% |
| Miausort v3 (in-place) | 127.94 | 3.40 | 2.25 | 100.0% |
| Wiki Sort (sqrt n) | 172.44 | 6.60 | 4.37 | 100.0% |
| Grailsort (sqrt n) | 300.24 | 6.04 | 4.61 | 100.0% |
| Wiki Sort (in-place) | 369.82 | 20.32 | 12.53 | 100.0% |
| Introsort | 439.31 | 10.75 | 6.99 | 100.0% |
| Grailsort (in-place) | 471.28 | 13.16 | 8.19 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **82.86** | 1.68 | 1.64 | 100.0% |
| Miausort v3 (sqrt n) | 160.00 | 6.00 | 4.88 | 100.0% |
| Timsort | 168.79 | 7.94 | 5.77 | 100.0% |
| Powersort | 179.93 | 7.05 | 5.26 | 100.0% |
| Wiki Sort (sqrt n) | 285.29 | 9.53 | 7.27 | 100.0% |
| Miausort v3 (in-place) | 315.92 | 11.77 | 9.05 | 100.0% |
| Grailsort (sqrt n) | 514.25 | 13.75 | 9.97 | 100.0% |
| Wiki Sort (in-place) | 633.86 | 17.65 | 14.03 | 100.0% |
| Grailsort (in-place) | 662.03 | 14.80 | 10.87 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **4.92** | 0.13 | 0.09 | 100.0% |
| Miausort v3 (in-place) | 5.01 | 0.17 | 0.11 | 100.0% |
| Powersort | 5.08 | 0.16 | 0.12 | 100.0% |
| Timsort | 5.12 | 0.14 | 0.13 | 100.0% |
| Introsort | 44.81 | 1.39 | 1.11 | 100.0% |
| Wiki Sort (sqrt n) | 52.47 | 2.08 | 1.44 | 100.0% |
| Wiki Sort (in-place) | 57.09 | 1.24 | 0.91 | 100.0% |
| Grailsort (sqrt n) | 129.04 | 2.97 | 2.83 | 100.0% |
| Grailsort (in-place) | 303.08 | 7.98 | 4.84 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **8.36** | 0.38 | 0.29 | 100.0% |
| Powersort | 8.53 | 0.19 | 0.22 | 100.0% |
| Miausort v3 (in-place) | 15.72 | 0.83 | 0.45 | 100.0% |
| Miausort v3 (sqrt n) | 15.98 | 0.89 | 0.76 | 100.0% |
| Introsort | 48.21 | 1.84 | 1.19 | 100.0% |
| Wiki Sort (sqrt n) | 157.81 | 4.17 | 2.37 | 100.0% |
| Wiki Sort (in-place) | 183.95 | 7.37 | 5.02 | 100.0% |
| Grailsort (sqrt n) | 233.40 | 5.81 | 5.29 | 100.0% |
| Grailsort (in-place) | 385.98 | 9.35 | 7.13 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **16.63** | 0.44 | 0.59 | 100.0% |
| Miausort v3 (in-place) | 16.75 | 0.44 | 0.32 | 100.0% |
| Introsort | 49.62 | 0.92 | 0.82 | 100.0% |
| Timsort | 171.53 | 3.21 | 2.47 | 100.0% |
| Powersort | 185.72 | 3.56 | 2.98 | 100.0% |
| Wiki Sort (sqrt n) | 199.76 | 6.15 | 3.91 | 100.0% |
| Wiki Sort (in-place) | 393.66 | 8.46 | 5.82 | 100.0% |
| Grailsort (sqrt n) | 396.88 | 7.94 | 5.60 | 100.0% |
| Grailsort (in-place) | 541.75 | 12.87 | 8.48 | 100.0% |
