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
| Introsort | **0.04** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.12 | 0.01 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 0.13 | 0.01 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 0.17 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.20 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.25 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.39 | 0.01 | 0.03 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.08 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.12 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.22 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.32 | 0.01 | 0.01 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.01** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.03 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.11 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.21 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.29 | 0.01 | 0.02 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.03 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.04 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.17 | 0.01 | 0.02 | 100.0% |
| Grailsort (in-place) | 0.19 | 0.02 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.23 | 0.01 | 0.01 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.02 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.03 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.11 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.19 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.26 | 0.02 | 0.02 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.03** | 0.00 | 0.00 | 100.0% |
| Introsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.03 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.10 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.10 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.19 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.28 | 0.01 | 0.02 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.06 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.08 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.08 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.21 | 0.01 | 0.02 | 100.0% |
| Grailsort (in-place) | 0.22 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.39 | 0.01 | 0.02 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.12 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.12 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.25 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.27 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.41 | 0.01 | 0.02 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.02** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.02 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.17 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.18 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.24 | 0.01 | 0.01 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.04 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.19 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.20 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.24 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.28 | 0.01 | 0.01 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.04** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.04 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.09 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.09 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.10 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.18 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.20 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.21 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.29 | 0.01 | 0.01 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.01 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.10 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.10 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.26 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.28 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.54 | 0.01 | 0.01 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.03** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.06 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.06 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.24 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.25 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.26 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 0.45 | 0.01 | 0.01 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.00** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.00 | 0.00 | 100.0% |
| Grailsort (in-place) | 0.10 | 0.00 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.13 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (in-place) | 0.19 | 0.01 | 0.00 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.01** | 0.00 | 0.00 | 100.0% |
| Powersort | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.00 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.16 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.17 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.23 | 0.04 | 0.02 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.01** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.01 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.07 | 0.00 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 0.22 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.28 | 0.01 | 0.01 | 100.0% |
| Grailsort (in-place) | 0.31 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 0.35 | 0.01 | 0.01 | 100.0% |

## List length 1000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.54** | 0.02 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.24 | 0.04 | 0.03 | 100.0% |
| Timsort | 1.26 | 0.04 | 0.02 | 100.0% |
| Powersort | 1.31 | 0.05 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 1.86 | 0.05 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 2.27 | 0.08 | 0.12 | 100.0% |
| Wiki Sort (sqrt n) | 2.62 | 0.12 | 0.08 | 100.0% |
| Grailsort (in-place) | 3.08 | 0.10 | 0.10 | 100.0% |
| Wiki Sort (in-place) | 4.53 | 0.14 | 0.10 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.45** | 0.01 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 0.74 | 0.01 | 0.02 | 100.0% |
| Timsort | 0.76 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.81 | 0.01 | 0.06 | 100.0% |
| Miausort v3 (in-place) | 0.94 | 0.02 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.20 | 0.04 | 0.06 | 100.0% |
| Grailsort (in-place) | 2.02 | 0.05 | 0.05 | 100.0% |
| Wiki Sort (sqrt n) | 2.17 | 0.03 | 0.05 | 100.0% |
| Wiki Sort (in-place) | 3.64 | 0.04 | 0.03 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.26** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.76 | 0.03 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 0.77 | 0.02 | 0.02 | 100.0% |
| Powersort | 0.81 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.11 | 0.03 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.68 | 0.08 | 0.07 | 100.0% |
| Wiki Sort (sqrt n) | 2.06 | 0.03 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.44 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 3.20 | 0.04 | 0.03 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.26** | 0.01 | 0.04 | 100.0% |
| Powersort | 0.26 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.28 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.47 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.47 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.29 | 0.04 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 1.69 | 0.04 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 1.72 | 0.03 | 0.06 | 100.0% |
| Grailsort (in-place) | 2.55 | 0.02 | 0.03 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.28** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.29 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.34 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.48 | 0.02 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.68 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.20 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.27 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 1.75 | 0.04 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.01 | 0.02 | 0.05 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.29** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.29 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.40 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.70 | 0.02 | 0.01 | 100.0% |
| Introsort | 1.24 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 1.26 | 0.03 | 0.04 | 100.0% |
| Grailsort (sqrt n) | 1.70 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 1.78 | 0.04 | 0.03 | 100.0% |
| Grailsort (in-place) | 2.52 | 0.03 | 0.03 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.42** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.97 | 0.05 | 0.03 | 100.0% |
| Powersort | 1.02 | 0.03 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 1.03 | 0.03 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.31 | 0.04 | 0.05 | 100.0% |
| Wiki Sort (sqrt n) | 2.27 | 0.07 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 2.72 | 0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 2.86 | 0.05 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 4.07 | 0.12 | 0.18 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.46** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 1.25 | 0.03 | 0.02 | 100.0% |
| Timsort | 1.26 | 0.03 | 0.05 | 100.0% |
| Powersort | 1.26 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 2.52 | 0.06 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 2.62 | 0.08 | 0.12 | 100.0% |
| Grailsort (sqrt n) | 3.46 | 0.07 | 0.06 | 100.0% |
| Grailsort (in-place) | 3.99 | 0.04 | 0.05 | 100.0% |
| Wiki Sort (in-place) | 4.53 | 0.21 | 0.19 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.18** | 0.01 | 0.01 | 100.0% |
| Timsort | 0.18 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.24 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.44 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 1.30 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 1.62 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 1.64 | 0.01 | 0.02 | 100.0% |
| Introsort | 1.84 | 0.03 | 0.04 | 100.0% |
| Grailsort (in-place) | 2.35 | 0.02 | 0.06 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.40** | 0.01 | 0.01 | 100.0% |
| Powersort | 0.46 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.51 | 0.01 | 0.01 | 100.0% |
| Introsort | 0.54 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.86 | 0.02 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 2.22 | 0.03 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 2.58 | 0.03 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 3.25 | 0.04 | 0.02 | 100.0% |
| Grailsort (in-place) | 3.46 | 0.07 | 0.05 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **0.59** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.60 | 0.01 | 0.01 | 100.0% |
| Powersort | 0.62 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.93 | 0.01 | 0.02 | 100.0% |
| Wiki Sort (sqrt n) | 1.69 | 0.01 | 0.02 | 100.0% |
| Introsort | 2.01 | 0.03 | 0.03 | 100.0% |
| Grailsort (sqrt n) | 2.09 | 0.02 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 2.71 | 0.03 | 0.09 | 100.0% |
| Grailsort (in-place) | 2.89 | 0.03 | 0.07 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.42** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 1.07 | 0.02 | 0.03 | 100.0% |
| Powersort | 1.17 | 0.03 | 0.02 | 100.0% |
| Timsort | 1.18 | 0.06 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 2.15 | 0.06 | 0.04 | 100.0% |
| Wiki Sort (sqrt n) | 2.54 | 0.07 | 0.07 | 100.0% |
| Grailsort (sqrt n) | 3.52 | 0.10 | 0.06 | 100.0% |
| Grailsort (in-place) | 4.06 | 0.07 | 0.13 | 100.0% |
| Wiki Sort (in-place) | 4.65 | 0.14 | 0.09 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **0.44** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.84 | 0.02 | 0.01 | 100.0% |
| Timsort | 0.84 | 0.02 | 0.06 | 100.0% |
| Powersort | 0.92 | 0.02 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 1.84 | 0.05 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 2.39 | 0.05 | 0.06 | 100.0% |
| Grailsort (sqrt n) | 3.18 | 0.03 | 0.02 | 100.0% |
| Grailsort (in-place) | 3.72 | 0.05 | 0.05 | 100.0% |
| Wiki Sort (in-place) | 4.81 | 0.19 | 0.12 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.04** | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.05 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.05 | 0.00 | 0.00 | 100.0% |
| Powersort | 0.05 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.22 | 0.01 | 0.01 | 100.0% |
| Wiki Sort (sqrt n) | 0.83 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 0.92 | 0.02 | 0.01 | 100.0% |
| Wiki Sort (in-place) | 1.00 | 0.02 | 0.01 | 100.0% |
| Grailsort (in-place) | 1.77 | 0.04 | 0.03 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.07** | 0.00 | 0.00 | 100.0% |
| Timsort | 0.07 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.00 | 0.02 | 100.0% |
| Introsort | 0.24 | 0.01 | 0.00 | 100.0% |
| Wiki Sort (sqrt n) | 1.56 | 0.13 | 0.06 | 100.0% |
| Wiki Sort (in-place) | 1.65 | 0.02 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 1.68 | 0.05 | 0.07 | 100.0% |
| Grailsort (in-place) | 2.33 | 0.02 | 0.03 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **0.14** | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.24 | 0.01 | 0.01 | 100.0% |
| Timsort | 1.08 | 0.01 | 0.01 | 100.0% |
| Powersort | 1.13 | 0.02 | 0.05 | 100.0% |
| Wiki Sort (sqrt n) | 1.85 | 0.02 | 0.02 | 100.0% |
| Grailsort (sqrt n) | 2.77 | 0.03 | 0.02 | 100.0% |
| Wiki Sort (in-place) | 3.06 | 0.04 | 0.02 | 100.0% |
| Grailsort (in-place) | 3.42 | 0.03 | 0.03 | 100.0% |

## List length 10000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **9.07** | 0.14 | 0.09 | 100.0% |
| Miausort v3 (sqrt n) | 15.24 | 0.30 | 0.31 | 100.0% |
| Timsort | 17.71 | 1.09 | 0.64 | 100.0% |
| Powersort | 18.37 | 1.02 | 0.51 | 100.0% |
| Miausort v3 (in-place) | 23.08 | 0.43 | 0.39 | 100.0% |
| Wiki Sort (sqrt n) | 24.81 | 0.77 | 0.82 | 100.0% |
| Grailsort (sqrt n) | 26.47 | 0.37 | 0.56 | 100.0% |
| Grailsort (in-place) | 40.41 | 0.61 | 1.18 | 100.0% |
| Wiki Sort (in-place) | 55.91 | 2.21 | 1.48 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **5.96** | 0.32 | 0.23 | 100.0% |
| Miausort v3 (sqrt n) | 8.47 | 0.07 | 0.05 | 100.0% |
| Timsort | 10.46 | 0.10 | 0.09 | 100.0% |
| Powersort | 11.78 | 0.20 | 0.35 | 100.0% |
| Miausort v3 (in-place) | 12.35 | 0.11 | 0.11 | 100.0% |
| Grailsort (sqrt n) | 14.20 | 0.07 | 0.13 | 100.0% |
| Wiki Sort (sqrt n) | 19.08 | 0.17 | 0.26 | 100.0% |
| Grailsort (in-place) | 26.99 | 0.25 | 0.40 | 100.0% |
| Wiki Sort (in-place) | 44.79 | 0.94 | 0.82 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **3.73** | 0.03 | 0.04 | 100.0% |
| Timsort | 9.89 | 0.08 | 0.12 | 100.0% |
| Miausort v3 (sqrt n) | 10.55 | 0.44 | 0.21 | 100.0% |
| Powersort | 10.95 | 0.05 | 0.16 | 100.0% |
| Miausort v3 (in-place) | 15.88 | 0.10 | 0.09 | 100.0% |
| Wiki Sort (sqrt n) | 19.86 | 0.11 | 0.46 | 100.0% |
| Grailsort (sqrt n) | 20.89 | 0.12 | 0.25 | 100.0% |
| Grailsort (in-place) | 34.61 | 1.41 | 0.72 | 100.0% |
| Wiki Sort (in-place) | 37.95 | 0.22 | 0.19 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **3.13** | 0.07 | 0.07 | 100.0% |
| Miausort v3 (sqrt n) | 3.17 | 0.04 | 0.03 | 100.0% |
| Powersort | 3.35 | 0.05 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 4.96 | 0.09 | 0.08 | 100.0% |
| Introsort | 6.05 | 0.06 | 0.08 | 100.0% |
| Wiki Sort (sqrt n) | 10.51 | 0.06 | 0.04 | 100.0% |
| Wiki Sort (in-place) | 16.43 | 0.10 | 0.08 | 100.0% |
| Grailsort (sqrt n) | 19.65 | 0.20 | 0.32 | 100.0% |
| Grailsort (in-place) | 33.69 | 0.16 | 0.29 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **3.10** | 0.03 | 0.03 | 100.0% |
| Timsort | 3.24 | 0.08 | 0.07 | 100.0% |
| Powersort | 3.39 | 0.05 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 6.29 | 0.12 | 0.12 | 100.0% |
| Introsort | 6.56 | 0.06 | 0.04 | 100.0% |
| Wiki Sort (sqrt n) | 10.38 | 0.07 | 0.24 | 100.0% |
| Grailsort (sqrt n) | 14.72 | 0.11 | 0.11 | 100.0% |
| Wiki Sort (in-place) | 15.74 | 0.10 | 0.07 | 100.0% |
| Grailsort (in-place) | 26.05 | 0.47 | 0.47 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **3.12** | 0.05 | 0.04 | 100.0% |
| Powersort | 3.12 | 0.05 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 3.49 | 0.04 | 0.03 | 100.0% |
| Miausort v3 (in-place) | 5.84 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (sqrt n) | 10.28 | 0.10 | 0.06 | 100.0% |
| Wiki Sort (in-place) | 14.83 | 0.08 | 0.07 | 100.0% |
| Introsort | 18.23 | 0.11 | 0.21 | 100.0% |
| Grailsort (sqrt n) | 19.21 | 0.13 | 0.20 | 100.0% |
| Grailsort (in-place) | 32.89 | 0.44 | 1.32 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **5.65** | 0.15 | 0.77 | 100.0% |
| Miausort v3 (sqrt n) | 11.05 | 0.11 | 0.10 | 100.0% |
| Timsort | 11.72 | 0.36 | 0.18 | 100.0% |
| Powersort | 12.76 | 0.07 | 0.25 | 100.0% |
| Miausort v3 (in-place) | 14.95 | 0.34 | 0.29 | 100.0% |
| Wiki Sort (sqrt n) | 19.27 | 0.17 | 0.18 | 100.0% |
| Grailsort (sqrt n) | 33.50 | 0.14 | 0.43 | 100.0% |
| Grailsort (in-place) | 34.97 | 0.61 | 0.40 | 100.0% |
| Wiki Sort (in-place) | 45.54 | 0.39 | 0.48 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **6.50** | 0.14 | 0.16 | 100.0% |
| Miausort v3 (sqrt n) | 14.01 | 0.11 | 0.21 | 100.0% |
| Timsort | 16.89 | 0.09 | 0.21 | 100.0% |
| Powersort | 18.10 | 0.09 | 0.09 | 100.0% |
| Wiki Sort (sqrt n) | 25.07 | 0.08 | 0.35 | 100.0% |
| Miausort v3 (in-place) | 28.61 | 1.42 | 0.81 | 100.0% |
| Grailsort (sqrt n) | 37.74 | 1.90 | 0.98 | 100.0% |
| Grailsort (in-place) | 47.04 | 1.90 | 1.03 | 100.0% |
| Wiki Sort (in-place) | 57.66 | 1.16 | 1.03 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **2.02** | 0.02 | 0.03 | 100.0% |
| Powersort | 2.03 | 0.03 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 2.64 | 0.05 | 0.06 | 100.0% |
| Miausort v3 (in-place) | 4.00 | 0.04 | 0.03 | 100.0% |
| Wiki Sort (sqrt n) | 11.71 | 0.08 | 0.16 | 100.0% |
| Wiki Sort (in-place) | 15.14 | 0.09 | 0.21 | 100.0% |
| Grailsort (sqrt n) | 18.88 | 0.22 | 0.15 | 100.0% |
| Grailsort (in-place) | 30.26 | 0.26 | 0.55 | 100.0% |
| Introsort | 33.12 | 0.20 | 0.50 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **4.11** | 0.11 | 0.09 | 100.0% |
| Powersort | 4.25 | 0.13 | 0.08 | 100.0% |
| Introsort | 7.15 | 0.12 | 0.08 | 100.0% |
| Miausort v3 (sqrt n) | 7.24 | 0.40 | 0.21 | 100.0% |
| Miausort v3 (in-place) | 11.83 | 0.16 | 0.16 | 100.0% |
| Wiki Sort (sqrt n) | 18.45 | 0.22 | 0.15 | 100.0% |
| Wiki Sort (in-place) | 28.52 | 0.74 | 0.66 | 100.0% |
| Grailsort (sqrt n) | 30.43 | 0.69 | 0.48 | 100.0% |
| Grailsort (in-place) | 43.80 | 1.79 | 1.10 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **6.82** | 0.06 | 0.04 | 100.0% |
| Timsort | 6.97 | 0.07 | 0.05 | 100.0% |
| Powersort | 7.61 | 0.07 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 10.44 | 0.22 | 0.15 | 100.0% |
| Wiki Sort (sqrt n) | 15.34 | 0.05 | 0.13 | 100.0% |
| Grailsort (sqrt n) | 24.52 | 1.35 | 0.71 | 100.0% |
| Introsort | 30.91 | 0.38 | 0.25 | 100.0% |
| Wiki Sort (in-place) | 31.13 | 0.10 | 0.10 | 100.0% |
| Grailsort (in-place) | 37.12 | 0.13 | 0.10 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **5.67** | 0.04 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 11.61 | 0.08 | 0.44 | 100.0% |
| Timsort | 13.17 | 0.10 | 0.09 | 100.0% |
| Powersort | 14.36 | 0.10 | 0.07 | 100.0% |
| Wiki Sort (sqrt n) | 22.67 | 0.20 | 0.78 | 100.0% |
| Miausort v3 (in-place) | 24.54 | 0.12 | 0.20 | 100.0% |
| Grailsort (sqrt n) | 40.40 | 1.22 | 0.66 | 100.0% |
| Grailsort (in-place) | 49.18 | 0.40 | 0.83 | 100.0% |
| Wiki Sort (in-place) | 53.88 | 1.90 | 1.56 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **5.78** | 0.04 | 0.04 | 100.0% |
| Miausort v3 (sqrt n) | 10.34 | 0.45 | 0.23 | 100.0% |
| Timsort | 12.47 | 0.08 | 0.08 | 100.0% |
| Powersort | 13.92 | 0.07 | 0.07 | 100.0% |
| Miausort v3 (in-place) | 22.26 | 0.11 | 0.11 | 100.0% |
| Wiki Sort (sqrt n) | 24.04 | 0.12 | 0.64 | 100.0% |
| Grailsort (sqrt n) | 36.25 | 0.30 | 0.27 | 100.0% |
| Grailsort (in-place) | 47.50 | 2.44 | 1.24 | 100.0% |
| Wiki Sort (in-place) | 55.44 | 0.79 | 0.41 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **0.49** | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.49 | 0.01 | 0.01 | 100.0% |
| Timsort | 0.50 | 0.02 | 0.01 | 100.0% |
| Powersort | 0.50 | 0.02 | 0.01 | 100.0% |
| Introsort | 3.54 | 0.04 | 0.04 | 100.0% |
| Wiki Sort (sqrt n) | 6.18 | 0.05 | 0.03 | 100.0% |
| Wiki Sort (in-place) | 7.16 | 0.08 | 0.05 | 100.0% |
| Grailsort (sqrt n) | 10.91 | 0.06 | 0.05 | 100.0% |
| Grailsort (in-place) | 24.22 | 0.86 | 0.52 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **0.79** | 0.02 | 0.02 | 100.0% |
| Timsort | 0.79 | 0.02 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.52 | 0.04 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 1.52 | 0.05 | 0.06 | 100.0% |
| Introsort | 3.74 | 0.10 | 0.08 | 100.0% |
| Wiki Sort (sqrt n) | 13.98 | 0.11 | 0.08 | 100.0% |
| Wiki Sort (in-place) | 17.20 | 0.64 | 0.35 | 100.0% |
| Grailsort (sqrt n) | 18.72 | 0.20 | 0.33 | 100.0% |
| Grailsort (in-place) | 29.35 | 0.16 | 0.23 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **1.62** | 0.02 | 0.02 | 100.0% |
| Miausort v3 (sqrt n) | 1.62 | 0.03 | 0.14 | 100.0% |
| Introsort | 4.01 | 0.06 | 0.66 | 100.0% |
| Timsort | 13.10 | 0.71 | 0.46 | 100.0% |
| Powersort | 14.48 | 0.06 | 0.08 | 100.0% |
| Wiki Sort (sqrt n) | 17.35 | 0.11 | 0.25 | 100.0% |
| Grailsort (sqrt n) | 32.46 | 1.19 | 0.60 | 100.0% |
| Wiki Sort (in-place) | 35.66 | 0.16 | 0.12 | 100.0% |
| Grailsort (in-place) | 43.51 | 1.65 | 0.87 | 100.0% |

## List length 100000

### gen_random

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **115.31** | 3.63 | 2.11 | 100.0% |
| Miausort v3 (sqrt n) | 189.29 | 1.95 | 3.08 | 100.0% |
| Timsort | 226.85 | 5.91 | 4.25 | 100.0% |
| Powersort | 227.96 | 3.02 | 4.51 | 100.0% |
| Wiki Sort (sqrt n) | 270.30 | 1.73 | 3.63 | 100.0% |
| Miausort v3 (in-place) | 300.03 | 10.82 | 5.98 | 100.0% |
| Grailsort (sqrt n) | 340.77 | 9.20 | 5.60 | 100.0% |
| Grailsort (in-place) | 473.70 | 13.85 | 9.03 | 100.0% |
| Wiki Sort (in-place) | 644.94 | 25.05 | 15.60 | 100.0% |

### gen_noisy

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **77.51** | 4.99 | 2.49 | 100.0% |
| Miausort v3 (sqrt n) | 103.33 | 4.38 | 2.30 | 100.0% |
| Timsort | 134.41 | 5.56 | 3.68 | 100.0% |
| Powersort | 150.36 | 2.30 | 2.52 | 100.0% |
| Miausort v3 (in-place) | 166.75 | 7.10 | 4.81 | 100.0% |
| Grailsort (sqrt n) | 178.37 | 9.07 | 4.88 | 100.0% |
| Wiki Sort (sqrt n) | 230.45 | 4.22 | 8.75 | 100.0% |
| Grailsort (in-place) | 344.25 | 10.45 | 14.94 | 100.0% |
| Wiki Sort (in-place) | 545.23 | 9.72 | 57.48 | 100.0% |

### gen_few_rand

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **56.32** | 0.86 | 1.22 | 100.0% |
| Miausort v3 (sqrt n) | 136.95 | 3.13 | 2.59 | 100.0% |
| Timsort | 137.70 | 2.13 | 2.65 | 100.0% |
| Powersort | 144.99 | 5.75 | 3.11 | 100.0% |
| Miausort v3 (in-place) | 235.82 | 7.02 | 6.83 | 100.0% |
| Wiki Sort (sqrt n) | 257.92 | 14.93 | 23.61 | 100.0% |
| Grailsort (sqrt n) | 273.94 | 11.74 | 7.29 | 100.0% |
| Grailsort (in-place) | 460.25 | 20.56 | 10.87 | 100.0% |
| Wiki Sort (in-place) | 499.57 | 58.76 | 81.75 | 100.0% |

### gen_scrambled_head

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **38.57** | 1.95 | 1.52 | 100.0% |
| Timsort | 41.68 | 2.18 | 1.39 | 100.0% |
| Powersort | 42.95 | 1.26 | 1.33 | 100.0% |
| Miausort v3 (in-place) | 61.98 | 4.33 | 3.03 | 100.0% |
| Introsort | 91.97 | 2.61 | 1.86 | 100.0% |
| Wiki Sort (sqrt n) | 107.41 | 2.36 | 2.05 | 100.0% |
| Wiki Sort (in-place) | 171.41 | 2.92 | 2.86 | 100.0% |
| Grailsort (sqrt n) | 250.82 | 11.64 | 13.86 | 100.0% |
| Grailsort (in-place) | 414.96 | 18.68 | 19.13 | 100.0% |

### gen_scrambled_tail

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **39.11** | 0.96 | 0.78 | 100.0% |
| Powersort | 41.02 | 1.51 | 0.91 | 100.0% |
| Miausort v3 (sqrt n) | 44.07 | 1.56 | 1.63 | 100.0% |
| Miausort v3 (in-place) | 84.04 | 3.41 | 2.00 | 100.0% |
| Wiki Sort (sqrt n) | 103.53 | 1.95 | 1.35 | 100.0% |
| Introsort | 104.57 | 2.33 | 1.62 | 100.0% |
| Wiki Sort (in-place) | 164.37 | 8.15 | 5.35 | 100.0% |
| Grailsort (sqrt n) | 185.29 | 9.58 | 5.75 | 100.0% |
| Grailsort (in-place) | 348.48 | 8.81 | 7.57 | 100.0% |

### gen_sawtooth

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **33.12** | 0.26 | 0.42 | 100.0% |
| Timsort | 33.23 | 0.35 | 0.30 | 100.0% |
| Miausort v3 (sqrt n) | 37.45 | 0.66 | 0.59 | 100.0% |
| Miausort v3 (in-place) | 60.83 | 1.28 | 1.17 | 100.0% |
| Wiki Sort (sqrt n) | 93.43 | 4.70 | 3.13 | 100.0% |
| Wiki Sort (in-place) | 140.91 | 4.02 | 4.97 | 100.0% |
| Grailsort (sqrt n) | 239.67 | 13.18 | 7.70 | 100.0% |
| Introsort | 311.18 | 8.16 | 6.11 | 100.0% |
| Grailsort (in-place) | 398.15 | 5.22 | 6.22 | 100.0% |

### gen_4_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **73.68** | 2.11 | 1.47 | 100.0% |
| Timsort | 135.75 | 4.08 | 3.30 | 100.0% |
| Miausort v3 (sqrt n) | 136.39 | 2.47 | 1.73 | 100.0% |
| Powersort | 148.24 | 5.91 | 3.37 | 100.0% |
| Miausort v3 (in-place) | 178.13 | 4.94 | 2.51 | 100.0% |
| Wiki Sort (sqrt n) | 227.13 | 2.16 | 2.26 | 100.0% |
| Grailsort (sqrt n) | 399.85 | 9.80 | 8.01 | 100.0% |
| Grailsort (in-place) | 413.31 | 6.07 | 10.34 | 100.0% |
| Wiki Sort (in-place) | 528.86 | 25.93 | 16.90 | 100.0% |

### gen_sqrtn_unique

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **80.84** | 1.76 | 1.78 | 100.0% |
| Miausort v3 (sqrt n) | 182.43 | 10.12 | 4.99 | 100.0% |
| Timsort | 214.65 | 6.04 | 4.54 | 100.0% |
| Powersort | 223.38 | 7.48 | 4.25 | 100.0% |
| Wiki Sort (sqrt n) | 282.42 | 7.55 | 3.92 | 100.0% |
| Miausort v3 (in-place) | 344.48 | 6.75 | 6.68 | 100.0% |
| Grailsort (sqrt n) | 466.95 | 6.30 | 4.10 | 100.0% |
| Grailsort (in-place) | 625.26 | 29.00 | 15.60 | 100.0% |
| Wiki Sort (in-place) | 698.20 | 32.35 | 16.79 | 100.0% |

### gen_pipe_organ

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Timsort | **19.46** | 0.13 | 0.12 | 100.0% |
| Powersort | 19.48 | 0.15 | 0.31 | 100.0% |
| Miausort v3 (sqrt n) | 25.64 | 0.16 | 0.29 | 100.0% |
| Miausort v3 (in-place) | 38.67 | 0.45 | 0.46 | 100.0% |
| Wiki Sort (sqrt n) | 112.50 | 0.61 | 0.75 | 100.0% |
| Wiki Sort (in-place) | 145.28 | 0.91 | 1.23 | 100.0% |
| Grailsort (sqrt n) | 217.50 | 3.04 | 4.86 | 100.0% |
| Grailsort (in-place) | 372.41 | 2.65 | 6.25 | 100.0% |
| Introsort | 473.44 | 7.83 | 10.88 | 100.0% |

### gen_grail_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **49.95** | 1.49 | 1.24 | 100.0% |
| Timsort | 50.74 | 1.73 | 0.87 | 100.0% |
| Miausort v3 (sqrt n) | 87.30 | 1.57 | 2.63 | 100.0% |
| Introsort | 93.12 | 1.50 | 0.99 | 100.0% |
| Miausort v3 (in-place) | 132.09 | 1.23 | 1.37 | 100.0% |
| Wiki Sort (sqrt n) | 192.55 | 10.88 | 5.01 | 100.0% |
| Wiki Sort (in-place) | 289.79 | 7.50 | 9.65 | 100.0% |
| Grailsort (sqrt n) | 360.70 | 6.57 | 4.78 | 100.0% |
| Grailsort (in-place) | 537.21 | 4.16 | 4.91 | 100.0% |

### gen_quick_adversary

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **80.82** | 3.80 | 2.12 | 100.0% |
| Timsort | 81.56 | 4.13 | 2.08 | 100.0% |
| Powersort | 83.84 | 0.41 | 0.96 | 100.0% |
| Miausort v3 (in-place) | 126.06 | 6.97 | 3.41 | 100.0% |
| Wiki Sort (sqrt n) | 162.68 | 2.74 | 2.32 | 100.0% |
| Grailsort (sqrt n) | 280.47 | 5.24 | 5.18 | 100.0% |
| Wiki Sort (in-place) | 343.83 | 18.33 | 8.88 | 100.0% |
| Introsort | 417.81 | 20.38 | 9.42 | 100.0% |
| Grailsort (in-place) | 448.23 | 26.33 | 12.40 | 100.0% |

### gen_triangular_heap

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **78.96** | 1.68 | 2.08 | 100.0% |
| Miausort v3 (sqrt n) | 141.75 | 1.70 | 1.21 | 100.0% |
| Timsort | 152.82 | 1.90 | 1.68 | 100.0% |
| Powersort | 163.47 | 2.29 | 2.30 | 100.0% |
| Wiki Sort (sqrt n) | 270.05 | 10.46 | 6.74 | 100.0% |
| Miausort v3 (in-place) | 280.19 | 1.57 | 1.72 | 100.0% |
| Grailsort (sqrt n) | 491.98 | 20.83 | 10.44 | 100.0% |
| Wiki Sort (in-place) | 598.82 | 23.71 | 12.45 | 100.0% |
| Grailsort (in-place) | 615.94 | 16.22 | 9.98 | 100.0% |

### gen_triangular

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Introsort | **83.20** | 2.04 | 1.16 | 100.0% |
| Miausort v3 (sqrt n) | 137.64 | 2.08 | 2.00 | 100.0% |
| Timsort | 161.90 | 4.82 | 5.37 | 100.0% |
| Powersort | 171.50 | 5.23 | 3.33 | 100.0% |
| Wiki Sort (sqrt n) | 265.42 | 4.67 | 3.03 | 100.0% |
| Miausort v3 (in-place) | 290.47 | 3.87 | 2.58 | 100.0% |
| Grailsort (sqrt n) | 462.20 | 16.79 | 10.16 | 100.0% |
| Grailsort (in-place) | 606.35 | 16.64 | 12.84 | 100.0% |
| Wiki Sort (in-place) | 619.07 | 24.20 | 17.62 | 100.0% |

### gen_sorted

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (sqrt n) | **4.61** | 0.07 | 0.06 | 100.0% |
| Timsort | 4.75 | 0.13 | 0.13 | 100.0% |
| Miausort v3 (in-place) | 4.80 | 0.23 | 0.17 | 100.0% |
| Powersort | 4.93 | 0.07 | 0.06 | 100.0% |
| Introsort | 44.85 | 0.39 | 0.35 | 100.0% |
| Wiki Sort (sqrt n) | 50.45 | 0.31 | 0.31 | 100.0% |
| Wiki Sort (in-place) | 57.39 | 0.29 | 0.52 | 100.0% |
| Grailsort (sqrt n) | 121.01 | 2.89 | 2.38 | 100.0% |
| Grailsort (in-place) | 286.83 | 13.83 | 6.98 | 100.0% |

### gen_reversed

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Powersort | **8.12** | 0.06 | 0.08 | 100.0% |
| Timsort | 8.18 | 0.33 | 0.23 | 100.0% |
| Miausort v3 (sqrt n) | 14.87 | 0.18 | 0.39 | 100.0% |
| Miausort v3 (in-place) | 15.00 | 0.22 | 0.24 | 100.0% |
| Introsort | 48.15 | 0.40 | 0.43 | 100.0% |
| Wiki Sort (sqrt n) | 146.35 | 1.31 | 1.14 | 100.0% |
| Wiki Sort (in-place) | 171.68 | 10.29 | 5.34 | 100.0% |
| Grailsort (sqrt n) | 217.96 | 13.33 | 6.43 | 100.0% |
| Grailsort (in-place) | 354.64 | 20.31 | 9.78 | 100.0% |

### gen_reversed_duplicates

| Algorithm | Median (ms) | IQR | stdev | Correct % |
|-----------|-------------|-----|-------|-----------|
| Miausort v3 (in-place) | **15.32** | 0.09 | 0.17 | 100.0% |
| Miausort v3 (sqrt n) | 16.04 | 0.22 | 0.25 | 100.0% |
| Introsort | 46.20 | 0.84 | 0.57 | 100.0% |
| Timsort | 153.55 | 8.59 | 4.50 | 100.0% |
| Powersort | 164.12 | 1.35 | 1.12 | 100.0% |
| Wiki Sort (sqrt n) | 181.62 | 1.75 | 2.07 | 100.0% |
| Wiki Sort (in-place) | 366.29 | 6.03 | 7.99 | 100.0% |
| Grailsort (sqrt n) | 376.30 | 8.19 | 9.77 | 100.0% |
| Grailsort (in-place) | 512.75 | 22.26 | 12.93 | 100.0% |
