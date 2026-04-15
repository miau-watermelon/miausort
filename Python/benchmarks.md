# Key

| Input | Description |
|-------|-------------|
| gen_random | Random shuffle of distinct integers 0 to n-1 |
| gen_noisy | Shuffle, each element only O(sqrt n) from its sorted position |
| gen_few_rand | Shuffle, only n/20 elements swapped |
| gen_scrambled_head | Shuffle, last 7/8ths sorted |
| gen_scrambled_tail | Shuffle, first 7/8ths sorted |
| gen_sawtooth | Four sorted runs |
| gen_4_unique | Shuffle of repeated integers 0 to 3 |
| gen_sqrtn_unique | Shuffle of repeated integers, exactly one less than Grail's/Miau's key target (~2 sqrt n - 1) |
| gen_pipe_organ | One ascending run followed by one descending run |
| gen_grail_adversary | Input designed to maximize Grailsort's buffer rewinds |
| gen_quick_adversary | Input designed to maximize bad pivot selection for median-of-3 |
| gen_triangular_heap | Heap based on triangular numbers on 2 sqrt n distinct values; chosen to maximize key collection work |
| gen_triangular | Permutation of 2 sqrt n distinct values based on population count of indices, maximizes merge and key collection work for Grail and Wiki (luckily, Miau is structurally immune) |
| gen_sorted | Sorted run |
| gen_reversed | Descending run with only distinct values |
| gen_reversed_duplicates | Descending run with n/2 distinct values (e.g. `[4, 4, 3, 3, 2, 2, 1, 1, 0, 0]` for n=10) |

# Benchmark (50 runs)

## List length 10

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
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
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.01 | 0.01 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.01 | 100.0% |

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
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.01 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
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
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.00** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_triangular

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

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
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
| Grailsort (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.02 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
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
| Powersort | 0.07 | 0.00 | 0.01 | 100.0% |
| Timsort | 0.07 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.01 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.19 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.21 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.25 | 0.01 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 0.38 | 0.01 | 0.01 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.01 | 0.02 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.10 | 0.01 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.13 | 0.01 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.23 | 0.01 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.34 | 0.01 | 0.01 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.10 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.14 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.24 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 0.29 | 0.01 | 0.01 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.03 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.13 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.16 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.18 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.24 | 0.01 | 0.01 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | **0.02** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.08 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.12 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.21 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.26 | 0.01 | 0.01 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.03** | 0.00 | 0.01 | 100.0% |
| Powersort | 0.03 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.09 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.11 | 0.02 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.01 | 0.02 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.20 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.29 | 0.02 | 0.01 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.06 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.08 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.09 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.21 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.24 | 0.04 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 0.27 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.39 | 0.01 | 0.05 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.13 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.13 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.24 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.28 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.44 | 0.01 | 0.03 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.02 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.18 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.18 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.24 | 0.01 | 0.04 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.04** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.19 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.21 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.23 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.37 | 0.08 | 0.04 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.04** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.09 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.09 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.10 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.18 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.21 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.21 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.30 | 0.01 | 0.01 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.11 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.11 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.29 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.48 | 0.01 | 0.01 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.06 | 0.00 | 0.01 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.08 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.08 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.24 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.25 | 0.01 | 0.02 | 100.0% |
| Grailsort (in-place) | 0.26 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.55 | 0.10 | 0.05 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.11 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.13 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.19 | 0.01 | 0.01 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.01** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.02 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.13 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.16 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.18 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.23 | 0.01 | 0.01 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.02 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.22 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.28 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.31 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.35 | 0.01 | 0.01 | 100.0% |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.74** | 0.02 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.28 | 0.03 | 0.03 | 100.0% |
| Timsort | 1.29 | 0.04 | 0.03 | 100.0% |
| Powersort | 1.34 | 0.04 | 0.04 | 100.0% |
| Miausort v3 (in-place) | 1.88 | 0.07 | 0.07 | 100.0% |
| Grailsort (sqrt n) | 2.22 | 0.05 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 2.65 | 0.10 | 0.06 | 100.0% |
| Grailsort (in-place) | 2.92 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 4.65 | 0.18 | 0.10 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.47** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.77 | 0.01 | 0.02 | 100.0% |
| Timsort | 0.80 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.85 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.97 | 0.01 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.20 | 0.03 | 0.05 | 100.0% |
| Grailsort (in-place) | 2.00 | 0.05 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 2.07 | 0.10 | 0.10 | 100.0% |
| Wiki Sort (in-place) | 3.64 | 0.04 | 0.05 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.28** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.73 | 0.01 | 0.02 | 100.0% |
| Timsort | 0.76 | 0.02 | 0.04 | 100.0% |
| Powersort | 0.81 | 0.02 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 1.06 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.61 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 2.02 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.40 | 0.05 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 3.15 | 0.11 | 0.06 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.26** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.28 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.29 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.49 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.49 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.28 | 0.02 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.71 | 0.24 | 0.12 | 100.0% |
| Grailsort (sqrt n) | 1.72 | 0.01 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.52 | 0.04 | 0.06 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.29** | 0.01 | 0.01 | 100.0% |
| Powersort | 0.30 | 0.01 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 0.34 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.53 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 0.69 | 0.01 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.30 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 1.31 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 1.82 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.01 | 0.05 | 0.06 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.29** | 0.01 | 0.01 | 100.0% |
| Powersort | 0.29 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.40 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.71 | 0.02 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.23 | 0.04 | 0.05 | 100.0% |
| Introsort | 1.27 | 0.04 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.69 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.78 | 0.04 | 0.05 | 100.0% |
| Grailsort (in-place) | 2.41 | 0.09 | 0.08 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.42** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.96 | 0.02 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.02 | 0.02 | 0.02 | 100.0% |
| Powersort | 1.02 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 1.33 | 0.04 | 0.07 | 100.0% |
| Wiki Sort (sqrt n) | 2.21 | 0.06 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 2.90 | 0.05 | 0.03 | 100.0% |
| Grailsort (in-place) | 3.01 | 0.06 | 0.10 | 100.0% |
| Wiki Sort (in-place) | 4.13 | 0.12 | 0.22 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.51** | 0.04 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.23 | 0.03 | 0.02 | 100.0% |
| Timsort | 1.26 | 0.03 | 0.03 | 100.0% |
| Powersort | 1.31 | 0.03 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 2.59 | 0.07 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 2.62 | 0.09 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 3.37 | 0.04 | 0.04 | 100.0% |
| Grailsort (in-place) | 3.89 | 0.04 | 0.05 | 100.0% |
| Wiki Sort (in-place) | 4.77 | 0.24 | 0.15 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.18** | 0.01 | 0.01 | 100.0% |
| Powersort | 0.18 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.24 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.44 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.32 | 0.03 | 0.07 | 100.0% |
| Grailsort (sqrt n) | 1.62 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 1.65 | 0.02 | 0.02 | 100.0% |
| Introsort | 1.83 | 0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.33 | 0.03 | 0.04 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.39** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.50 | 0.01 | 0.02 | 100.0% |
| Powersort | 0.53 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.62 | 0.02 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 0.85 | 0.04 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 2.18 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 2.49 | 0.24 | 0.12 | 100.0% |
| Wiki Sort (in-place) | 3.16 | 0.04 | 0.06 | 100.0% |
| Grailsort (in-place) | 3.33 | 0.05 | 0.05 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.62** | 0.01 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 0.63 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.65 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.97 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.77 | 0.03 | 0.03 | 100.0% |
| Introsort | 2.08 | 0.03 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 2.13 | 0.06 | 0.05 | 100.0% |
| Wiki Sort (in-place) | 2.84 | 0.05 | 0.04 | 100.0% |
| Grailsort (in-place) | 2.92 | 0.09 | 0.11 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.44** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 1.08 | 0.02 | 0.02 | 100.0% |
| Timsort | 1.17 | 0.02 | 0.02 | 100.0% |
| Powersort | 1.22 | 0.04 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 2.27 | 0.04 | 0.07 | 100.0% |
| Wiki Sort (sqrt n) | 2.60 | 0.06 | 0.09 | 100.0% |
| Grailsort (sqrt n) | 3.66 | 0.08 | 0.05 | 100.0% |
| Grailsort (in-place) | 4.06 | 0.06 | 0.26 | 100.0% |
| Wiki Sort (in-place) | 5.01 | 0.16 | 0.19 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.45** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.88 | 0.11 | 0.06 | 100.0% |
| Powersort | 0.94 | 0.02 | 0.01 | 100.0% |
| Timsort | 0.94 | 0.03 | 0.04 | 100.0% |
| Miausort v3 (in-place) | 1.90 | 0.05 | 0.10 | 100.0% |
| Wiki Sort (sqrt n) | 2.50 | 0.08 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 3.25 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 3.78 | 0.04 | 0.06 | 100.0% |
| Wiki Sort (in-place) | 4.92 | 0.13 | 0.10 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.04** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.23 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.80 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.90 | 0.07 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.12 | 0.15 | 0.07 | 100.0% |
| Grailsort (in-place) | 1.69 | 0.03 | 0.03 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.07** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.08 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.15 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.15 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.26 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.47 | 0.08 | 0.11 | 100.0% |
| Grailsort (sqrt n) | 1.70 | 0.04 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 1.74 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.39 | 0.03 | 0.02 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.15** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.15 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.25 | 0.01 | 0.01 | 100.0% |
| Timsort | 1.12 | 0.01 | 0.01 | 100.0% |
| Powersort | 1.17 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.87 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 2.83 | 0.02 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 3.11 | 0.02 | 0.23 | 100.0% |
| Grailsort (in-place) | 3.49 | 0.04 | 0.04 | 100.0% |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **12.08** | 0.63 | 0.37 | 100.0% |
| Miausort v3 (sqrt n) | 14.15 | 0.10 | 0.11 | 100.0% |
| Timsort | 17.16 | 0.18 | 0.30 | 100.0% |
| Powersort | 18.95 | 0.23 | 0.17 | 100.0% |
| Miausort v3 (in-place) | 22.63 | 0.11 | 0.32 | 100.0% |
| Wiki Sort (sqrt n) | 24.96 | 0.12 | 0.12 | 100.0% |
| Grailsort (sqrt n) | 25.91 | 0.13 | 0.10 | 100.0% |
| Grailsort (in-place) | 36.97 | 1.54 | 1.09 | 100.0% |
| Wiki Sort (in-place) | 57.85 | 1.13 | 0.67 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.34** | 0.08 | 0.06 | 100.0% |
| Miausort v3 (sqrt n) | 9.09 | 0.05 | 0.05 | 100.0% |
| Timsort | 10.73 | 0.05 | 0.05 | 100.0% |
| Powersort | 12.09 | 0.14 | 0.26 | 100.0% |
| Miausort v3 (in-place) | 12.95 | 0.11 | 0.09 | 100.0% |
| Grailsort (sqrt n) | 14.44 | 0.10 | 0.12 | 100.0% |
| Wiki Sort (sqrt n) | 19.64 | 0.36 | 0.35 | 100.0% |
| Grailsort (in-place) | 26.40 | 0.14 | 0.14 | 100.0% |
| Wiki Sort (in-place) | 45.45 | 0.92 | 0.73 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.40** | 0.05 | 0.08 | 100.0% |
| Miausort v3 (sqrt n) | 10.05 | 0.21 | 0.18 | 100.0% |
| Timsort | 10.20 | 0.28 | 0.17 | 100.0% |
| Powersort | 11.46 | 0.27 | 0.25 | 100.0% |
| Miausort v3 (in-place) | 16.37 | 0.42 | 0.28 | 100.0% |
| Grailsort (sqrt n) | 20.66 | 0.16 | 0.34 | 100.0% |
| Wiki Sort (sqrt n) | 21.06 | 0.62 | 0.43 | 100.0% |
| Grailsort (in-place) | 32.91 | 0.28 | 0.49 | 100.0% |
| Wiki Sort (in-place) | 38.92 | 1.48 | 0.84 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **3.06** | 0.03 | 0.06 | 100.0% |
| Timsort | 3.17 | 0.08 | 0.06 | 100.0% |
| Powersort | 3.43 | 0.10 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 5.06 | 0.15 | 0.11 | 100.0% |
| Introsort | 6.84 | 0.10 | 0.14 | 100.0% |
| Wiki Sort (sqrt n) | 10.75 | 0.13 | 0.30 | 100.0% |
| Wiki Sort (in-place) | 16.65 | 0.11 | 0.35 | 100.0% |
| Grailsort (sqrt n) | 20.22 | 0.10 | 0.12 | 100.0% |
| Grailsort (in-place) | 32.75 | 0.27 | 0.40 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **3.29** | 0.04 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 3.37 | 0.04 | 0.06 | 100.0% |
| Powersort | 3.43 | 0.07 | 0.13 | 100.0% |
| Introsort | 6.60 | 0.10 | 0.10 | 100.0% |
| Miausort v3 (in-place) | 6.92 | 0.05 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 10.10 | 0.05 | 0.28 | 100.0% |
| Grailsort (sqrt n) | 13.69 | 0.07 | 0.08 | 100.0% |
| Wiki Sort (in-place) | 15.80 | 0.06 | 0.07 | 100.0% |
| Grailsort (in-place) | 25.83 | 0.79 | 0.43 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **3.26** | 0.06 | 0.04 | 100.0% |
| Powersort | 3.29 | 0.05 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 3.35 | 0.06 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 5.99 | 0.13 | 0.09 | 100.0% |
| Wiki Sort (sqrt n) | 10.42 | 0.19 | 0.14 | 100.0% |
| Wiki Sort (in-place) | 14.81 | 0.17 | 0.21 | 100.0% |
| Introsort | 18.57 | 0.29 | 0.28 | 100.0% |
| Grailsort (sqrt n) | 19.45 | 0.73 | 0.43 | 100.0% |
| Grailsort (in-place) | 32.19 | 0.15 | 0.20 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **5.92** | 0.06 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 11.43 | 0.10 | 0.07 | 100.0% |
| Timsort | 12.05 | 0.07 | 0.14 | 100.0% |
| Powersort | 13.36 | 0.10 | 0.09 | 100.0% |
| Miausort v3 (in-place) | 14.94 | 0.09 | 0.14 | 100.0% |
| Wiki Sort (sqrt n) | 20.45 | 0.10 | 0.08 | 100.0% |
| Grailsort (sqrt n) | 34.65 | 0.37 | 0.59 | 100.0% |
| Grailsort (in-place) | 36.07 | 1.36 | 0.83 | 100.0% |
| Wiki Sort (in-place) | 47.75 | 1.34 | 1.15 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.84** | 0.06 | 0.06 | 100.0% |
| Miausort v3 (sqrt n) | 13.85 | 0.05 | 0.05 | 100.0% |
| Timsort | 16.28 | 0.09 | 0.14 | 100.0% |
| Powersort | 17.37 | 0.06 | 0.07 | 100.0% |
| Wiki Sort (sqrt n) | 24.41 | 0.10 | 0.09 | 100.0% |
| Miausort v3 (in-place) | 26.68 | 0.48 | 0.43 | 100.0% |
| Grailsort (sqrt n) | 38.41 | 1.22 | 0.63 | 100.0% |
| Grailsort (in-place) | 48.23 | 0.26 | 0.34 | 100.0% |
| Wiki Sort (in-place) | 57.16 | 0.75 | 1.07 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **1.93** | 0.04 | 0.13 | 100.0% |
| Powersort | 1.96 | 0.10 | 0.06 | 100.0% |
| Miausort v3 (sqrt n) | 2.47 | 0.05 | 0.04 | 100.0% |
| Miausort v3 (in-place) | 4.13 | 0.05 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 11.73 | 0.06 | 0.10 | 100.0% |
| Wiki Sort (in-place) | 15.05 | 0.43 | 0.30 | 100.0% |
| Grailsort (sqrt n) | 18.66 | 0.55 | 0.36 | 100.0% |
| Grailsort (in-place) | 29.58 | 0.74 | 0.50 | 100.0% |
| Introsort | 34.63 | 1.04 | 0.54 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **4.01** | 0.05 | 0.05 | 100.0% |
| Powersort | 4.19 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 6.63 | 0.20 | 0.10 | 100.0% |
| Introsort | 8.43 | 0.55 | 0.30 | 100.0% |
| Miausort v3 (in-place) | 10.97 | 0.40 | 0.56 | 100.0% |
| Wiki Sort (sqrt n) | 18.67 | 0.08 | 0.08 | 100.0% |
| Wiki Sort (in-place) | 28.07 | 0.26 | 0.29 | 100.0% |
| Grailsort (sqrt n) | 31.02 | 0.31 | 0.31 | 100.0% |
| Grailsort (in-place) | 44.97 | 0.22 | 0.17 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **6.99** | 0.04 | 0.05 | 100.0% |
| Timsort | 7.19 | 0.16 | 0.52 | 100.0% |
| Powersort | 7.58 | 0.07 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 10.93 | 0.06 | 0.05 | 100.0% |
| Wiki Sort (sqrt n) | 15.88 | 0.07 | 0.06 | 100.0% |
| Grailsort (sqrt n) | 23.53 | 0.43 | 0.29 | 100.0% |
| Introsort | 30.93 | 0.31 | 0.46 | 100.0% |
| Wiki Sort (in-place) | 32.17 | 0.15 | 0.13 | 100.0% |
| Grailsort (in-place) | 35.50 | 0.82 | 0.70 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **5.88** | 0.05 | 0.11 | 100.0% |
| Miausort v3 (sqrt n) | 11.46 | 0.16 | 0.18 | 100.0% |
| Timsort | 13.29 | 0.06 | 0.24 | 100.0% |
| Powersort | 14.44 | 0.10 | 0.20 | 100.0% |
| Miausort v3 (in-place) | 23.12 | 0.13 | 0.33 | 100.0% |
| Wiki Sort (sqrt n) | 23.75 | 0.10 | 0.55 | 100.0% |
| Grailsort (sqrt n) | 40.44 | 0.94 | 0.66 | 100.0% |
| Grailsort (in-place) | 49.72 | 0.23 | 0.34 | 100.0% |
| Wiki Sort (in-place) | 53.51 | 1.93 | 1.02 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.08** | 0.06 | 0.09 | 100.0% |
| Miausort v3 (sqrt n) | 12.08 | 0.10 | 0.06 | 100.0% |
| Timsort | 13.24 | 0.20 | 0.36 | 100.0% |
| Powersort | 14.61 | 0.19 | 0.23 | 100.0% |
| Wiki Sort (sqrt n) | 24.78 | 0.46 | 0.30 | 100.0% |
| Miausort v3 (in-place) | 24.83 | 0.07 | 0.23 | 100.0% |
| Grailsort (sqrt n) | 37.00 | 0.69 | 0.34 | 100.0% |
| Grailsort (in-place) | 46.08 | 0.72 | 0.41 | 100.0% |
| Wiki Sort (in-place) | 57.84 | 1.29 | 1.31 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.48** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.48 | 0.02 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.49 | 0.02 | 0.01 | 100.0% |
| Powersort | 0.49 | 0.02 | 0.01 | 100.0% |
| Introsort | 3.66 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (sqrt n) | 6.17 | 0.05 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 7.17 | 0.06 | 0.16 | 100.0% |
| Grailsort (sqrt n) | 10.70 | 0.06 | 0.05 | 100.0% |
| Grailsort (in-place) | 23.41 | 0.13 | 0.10 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.78** | 0.02 | 0.02 | 100.0% |
| Powersort | 0.79 | 0.03 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.51 | 0.06 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 1.52 | 0.06 | 0.04 | 100.0% |
| Introsort | 3.95 | 0.06 | 0.06 | 100.0% |
| Wiki Sort (sqrt n) | 14.30 | 0.48 | 0.25 | 100.0% |
| Wiki Sort (in-place) | 17.51 | 0.20 | 0.14 | 100.0% |
| Grailsort (sqrt n) | 18.32 | 0.61 | 0.58 | 100.0% |
| Grailsort (in-place) | 29.64 | 0.20 | 0.28 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **1.55** | 0.03 | 0.04 | 100.0% |
| Miausort v3 (in-place) | 1.55 | 0.02 | 0.03 | 100.0% |
| Introsort | 3.90 | 0.07 | 0.06 | 100.0% |
| Timsort | 12.70 | 0.27 | 0.24 | 100.0% |
| Powersort | 13.99 | 0.38 | 0.24 | 100.0% |
| Wiki Sort (sqrt n) | 16.86 | 0.21 | 0.20 | 100.0% |
| Grailsort (sqrt n) | 31.15 | 1.30 | 0.71 | 100.0% |
| Wiki Sort (in-place) | 34.93 | 0.85 | 0.74 | 100.0% |
| Grailsort (in-place) | 41.96 | 1.32 | 0.72 | 100.0% |

## List length 100000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **106.78** | 3.74 | 4.36 | 100.0% |
| Miausort v3 (sqrt n) | 181.58 | 5.50 | 4.45 | 100.0% |
| Timsort | 222.51 | 6.18 | 3.44 | 100.0% |
| Powersort | 235.18 | 5.88 | 3.33 | 100.0% |
| Wiki Sort (sqrt n) | 280.11 | 5.89 | 3.48 | 100.0% |
| Miausort v3 (in-place) | 295.81 | 8.57 | 4.69 | 100.0% |
| Grailsort (sqrt n) | 326.05 | 9.03 | 4.91 | 100.0% |
| Grailsort (in-place) | 477.15 | 11.92 | 6.65 | 100.0% |
| Wiki Sort (in-place) | 656.28 | 11.08 | 9.33 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **81.53** | 4.44 | 2.77 | 100.0% |
| Miausort v3 (sqrt n) | 100.56 | 0.88 | 1.15 | 100.0% |
| Timsort | 132.31 | 1.85 | 1.72 | 100.0% |
| Powersort | 145.18 | 1.56 | 1.77 | 100.0% |
| Grailsort (sqrt n) | 165.13 | 4.88 | 3.67 | 100.0% |
| Miausort v3 (in-place) | 165.74 | 3.18 | 2.82 | 100.0% |
| Wiki Sort (sqrt n) | 212.42 | 4.95 | 4.03 | 100.0% |
| Grailsort (in-place) | 340.12 | 4.52 | 5.94 | 100.0% |
| Wiki Sort (in-place) | 525.75 | 22.98 | 16.19 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **74.01** | 0.58 | 0.65 | 100.0% |
| Miausort v3 (sqrt n) | 128.16 | 3.22 | 2.13 | 100.0% |
| Timsort | 129.35 | 3.46 | 2.43 | 100.0% |
| Powersort | 140.95 | 4.65 | 2.73 | 100.0% |
| Miausort v3 (in-place) | 217.00 | 4.75 | 3.48 | 100.0% |
| Wiki Sort (sqrt n) | 232.37 | 8.07 | 4.72 | 100.0% |
| Grailsort (sqrt n) | 266.06 | 4.53 | 4.12 | 100.0% |
| Grailsort (in-place) | 419.11 | 12.43 | 6.88 | 100.0% |
| Wiki Sort (in-place) | 476.52 | 19.34 | 11.30 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **36.89** | 0.20 | 0.33 | 100.0% |
| Timsort | 37.56 | 1.20 | 1.06 | 100.0% |
| Powersort | 39.48 | 0.95 | 0.60 | 100.0% |
| Miausort v3 (in-place) | 56.44 | 1.87 | 1.06 | 100.0% |
| Wiki Sort (sqrt n) | 103.21 | 0.65 | 0.65 | 100.0% |
| Introsort | 108.31 | 3.37 | 1.73 | 100.0% |
| Wiki Sort (in-place) | 172.23 | 5.22 | 3.54 | 100.0% |
| Grailsort (sqrt n) | 226.98 | 5.74 | 3.85 | 100.0% |
| Grailsort (in-place) | 391.33 | 13.48 | 6.63 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **37.54** | 0.44 | 0.41 | 100.0% |
| Miausort v3 (sqrt n) | 37.64 | 1.64 | 0.90 | 100.0% |
| Powersort | 39.97 | 0.55 | 0.68 | 100.0% |
| Miausort v3 (in-place) | 72.82 | 0.67 | 1.27 | 100.0% |
| Introsort | 92.40 | 2.14 | 1.33 | 100.0% |
| Wiki Sort (sqrt n) | 94.73 | 1.32 | 1.46 | 100.0% |
| Wiki Sort (in-place) | 155.74 | 3.85 | 2.22 | 100.0% |
| Grailsort (sqrt n) | 166.90 | 7.01 | 3.73 | 100.0% |
| Grailsort (in-place) | 331.13 | 10.67 | 5.84 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **31.86** | 0.32 | 0.30 | 100.0% |
| Timsort | 32.86 | 0.36 | 0.41 | 100.0% |
| Miausort v3 (sqrt n) | 33.75 | 0.28 | 0.24 | 100.0% |
| Miausort v3 (in-place) | 56.89 | 1.21 | 1.33 | 100.0% |
| Wiki Sort (sqrt n) | 88.71 | 0.62 | 0.82 | 100.0% |
| Wiki Sort (in-place) | 137.14 | 3.48 | 2.03 | 100.0% |
| Grailsort (sqrt n) | 223.53 | 8.54 | 4.28 | 100.0% |
| Introsort | 293.67 | 7.04 | 4.20 | 100.0% |
| Grailsort (in-place) | 379.37 | 7.47 | 5.91 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **76.50** | 0.88 | 0.61 | 100.0% |
| Miausort v3 (sqrt n) | 127.95 | 4.79 | 2.59 | 100.0% |
| Timsort | 132.43 | 3.77 | 2.85 | 100.0% |
| Powersort | 148.56 | 3.03 | 2.84 | 100.0% |
| Miausort v3 (in-place) | 179.30 | 5.14 | 3.18 | 100.0% |
| Wiki Sort (sqrt n) | 223.30 | 4.08 | 3.45 | 100.0% |
| Grailsort (sqrt n) | 388.86 | 9.57 | 8.42 | 100.0% |
| Grailsort (in-place) | 412.94 | 9.64 | 6.25 | 100.0% |
| Wiki Sort (in-place) | 520.61 | 24.48 | 14.43 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **89.34** | 2.95 | 1.49 | 100.0% |
| Miausort v3 (sqrt n) | 176.38 | 6.88 | 3.57 | 100.0% |
| Timsort | 213.81 | 6.66 | 3.76 | 100.0% |
| Powersort | 225.41 | 13.72 | 8.42 | 100.0% |
| Wiki Sort (sqrt n) | 281.80 | 9.98 | 6.72 | 100.0% |
| Miausort v3 (in-place) | 306.09 | 7.85 | 6.67 | 100.0% |
| Grailsort (sqrt n) | 454.15 | 4.29 | 4.35 | 100.0% |
| Grailsort (in-place) | 589.49 | 8.08 | 6.47 | 100.0% |
| Wiki Sort (in-place) | 696.75 | 25.23 | 15.22 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **19.28** | 0.21 | 0.38 | 100.0% |
| Powersort | 19.32 | 0.21 | 0.23 | 100.0% |
| Miausort v3 (sqrt n) | 25.75 | 1.19 | 0.64 | 100.0% |
| Miausort v3 (in-place) | 40.97 | 0.19 | 0.59 | 100.0% |
| Wiki Sort (sqrt n) | 112.10 | 1.46 | 1.03 | 100.0% |
| Wiki Sort (in-place) | 149.34 | 5.59 | 2.83 | 100.0% |
| Grailsort (sqrt n) | 218.12 | 8.39 | 4.51 | 100.0% |
| Grailsort (in-place) | 368.62 | 4.10 | 3.67 | 100.0% |
| Introsort | 462.73 | 7.08 | 4.76 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **47.87** | 0.55 | 0.66 | 100.0% |
| Timsort | 48.19 | 1.41 | 0.91 | 100.0% |
| Miausort v3 (sqrt n) | 76.18 | 2.34 | 1.49 | 100.0% |
| Introsort | 94.34 | 3.14 | 1.59 | 100.0% |
| Miausort v3 (in-place) | 116.52 | 1.87 | 1.96 | 100.0% |
| Wiki Sort (sqrt n) | 193.56 | 1.89 | 2.70 | 100.0% |
| Wiki Sort (in-place) | 292.26 | 7.80 | 4.02 | 100.0% |
| Grailsort (sqrt n) | 363.23 | 5.84 | 6.42 | 100.0% |
| Grailsort (in-place) | 536.03 | 7.42 | 8.15 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **79.41** | 3.49 | 1.85 | 100.0% |
| Miausort v3 (sqrt n) | 81.54 | 1.18 | 1.23 | 100.0% |
| Powersort | 85.69 | 3.91 | 2.00 | 100.0% |
| Miausort v3 (in-place) | 124.34 | 5.75 | 2.60 | 100.0% |
| Wiki Sort (sqrt n) | 168.20 | 1.88 | 1.72 | 100.0% |
| Grailsort (sqrt n) | 286.43 | 9.86 | 4.99 | 100.0% |
| Wiki Sort (in-place) | 351.50 | 11.81 | 6.34 | 100.0% |
| Introsort | 409.06 | 13.54 | 8.48 | 100.0% |
| Grailsort (in-place) | 450.50 | 4.43 | 5.99 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **77.96** | 1.48 | 0.96 | 100.0% |
| Miausort v3 (sqrt n) | 139.07 | 4.91 | 2.47 | 100.0% |
| Timsort | 156.08 | 4.40 | 2.53 | 100.0% |
| Powersort | 168.51 | 3.02 | 2.08 | 100.0% |
| Wiki Sort (sqrt n) | 258.10 | 4.74 | 3.33 | 100.0% |
| Miausort v3 (in-place) | 269.01 | 8.01 | 4.56 | 100.0% |
| Grailsort (sqrt n) | 489.34 | 5.52 | 6.45 | 100.0% |
| Wiki Sort (in-place) | 590.68 | 18.22 | 11.58 | 100.0% |
| Grailsort (in-place) | 616.02 | 14.72 | 9.23 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **84.30** | 2.22 | 1.26 | 100.0% |
| Miausort v3 (sqrt n) | 156.16 | 4.09 | 2.46 | 100.0% |
| Timsort | 160.86 | 1.89 | 2.31 | 100.0% |
| Powersort | 174.77 | 1.60 | 2.05 | 100.0% |
| Wiki Sort (sqrt n) | 267.15 | 1.87 | 8.38 | 100.0% |
| Miausort v3 (in-place) | 310.55 | 8.02 | 4.62 | 100.0% |
| Grailsort (sqrt n) | 437.36 | 7.20 | 4.65 | 100.0% |
| Grailsort (in-place) | 575.53 | 11.45 | 10.02 | 100.0% |
| Wiki Sort (in-place) | 608.13 | 15.00 | 8.45 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **4.65** | 0.13 | 0.11 | 100.0% |
| Miausort v3 (sqrt n) | 4.68 | 0.08 | 0.10 | 100.0% |
| Timsort | 4.79 | 0.13 | 0.12 | 100.0% |
| Powersort | 4.84 | 0.13 | 0.12 | 100.0% |
| Introsort | 43.38 | 0.51 | 0.44 | 100.0% |
| Wiki Sort (sqrt n) | 53.55 | 0.68 | 0.94 | 100.0% |
| Wiki Sort (in-place) | 54.79 | 0.66 | 0.58 | 100.0% |
| Grailsort (sqrt n) | 122.28 | 5.29 | 3.32 | 100.0% |
| Grailsort (in-place) | 283.12 | 5.17 | 6.67 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **7.93** | 0.10 | 0.08 | 100.0% |
| Powersort | 7.94 | 0.10 | 0.06 | 100.0% |
| Miausort v3 (in-place) | 15.05 | 0.19 | 0.19 | 100.0% |
| Miausort v3 (sqrt n) | 15.60 | 0.17 | 0.20 | 100.0% |
| Introsort | 45.58 | 0.28 | 0.35 | 100.0% |
| Wiki Sort (sqrt n) | 146.96 | 2.82 | 2.30 | 100.0% |
| Wiki Sort (in-place) | 173.03 | 2.89 | 1.87 | 100.0% |
| Grailsort (sqrt n) | 218.13 | 5.60 | 3.68 | 100.0% |
| Grailsort (in-place) | 352.12 | 15.39 | 7.60 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **15.52** | 0.13 | 0.19 | 100.0% |
| Miausort v3 (sqrt n) | 15.96 | 0.32 | 0.27 | 100.0% |
| Introsort | 46.62 | 1.62 | 1.05 | 100.0% |
| Timsort | 155.07 | 2.35 | 1.31 | 100.0% |
| Powersort | 168.25 | 4.53 | 3.08 | 100.0% |
| Wiki Sort (sqrt n) | 183.75 | 1.87 | 2.00 | 100.0% |
| Grailsort (sqrt n) | 373.83 | 5.11 | 6.62 | 100.0% |
| Wiki Sort (in-place) | 380.50 | 13.72 | 7.44 | 100.0% |
| Grailsort (in-place) | 508.75 | 10.40 | 9.45 | 100.0% |
