# Miau Sort
A stable, in-place O(n log n) worst-case adaptive sort taking ideas from [Grail Sort](https://github.com/Mrrl/GrailSort).
In development, but currently has the following features:
- Run scans with stable reversals
- Early exits in event of run monotony
- O(n) worst-case key collection (as of v3), thanks to bzy for O(n) run repair from dustsort (also means that this is now comparison optimal, doing `n log n + O(n)` comparisons in the worst-case) 
- O(sqrt n) block tag sorting, improvement on Grail's O(n) for this
- Significantly optimised key redistribution
- Rewritten merge as of v3 (from [Adaptive Grail Sort](https://github.com/Gaming32/ArrayV/blob/main/src/main/java/io/github/arrayv/sorts/hybrid/AdaptiveGrailSort.java))

Currently working on:
- Reimplementing galloping for v3 (this will likely significantly reduce performance on random data, though I will try to prevent this)
- Porting to more practical languages (Java, C, Rust)

Acknowledgements:
- Amari [(double-a git)](https://git.a-a.dev/amari) for ideas from Helium Sort and [UniV](https://git.a-a.dev/amari/UniV), the visualizer I used for the vast majority of this algorithm's development.
- The Holy Grailsort team [(github)](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file) for improved key sorting, rotations and block selection sort
- bzy ([github](https://github.com/bzyjin), [codeberg](https://codeberg.org/bzy/))  for help with optimised key collection (ideas from [dustsort](https://github.com/bzyjin/dustsort/tree/main)) and also help with debugging

# Python benchmarks:
(I tried to make it accurate, but keep in mind that Python isn't the most reliable)
(Edit: It was in microseconds before, I accidentally did the conversion twice...)

## Mean of 50 runs (gen_random)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.19 | 0.19 | 0.01 | 0.01 | 0.19 | 0.21 | 100.0% |
| Grailsort (sqrt n) | 0.17 | 0.18 | 0.01 | 0.01 | 0.16 | 0.24 | 100.0% |
| Miausort v2 (in-place) | 0.17 | 0.17 | 0.00 | 0.00 | 0.17 | 0.19 | 100.0% |
| Miausort v2 (sqrt n) | 0.17 | 0.17 | 0.00 | 0.00 | 0.17 | 0.19 | 100.0% |
| Miausort v3 (in-place) | 0.12 | 0.13 | 0.01 | 0.00 | 0.12 | 0.19 | 100.0% |
| Miausort v3 (sqrt n) | 0.12 | 0.13 | 0.00 | 0.00 | 0.12 | 0.14 | 100.0% |
| Timsort | 0.07 | 0.07 | 0.00 | 0.00 | 0.07 | 0.08 | 100.0% |
| Introsort | 0.03 | 0.03 | 0.00 | 0.00 | 0.03 | 0.04 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 2.87 | 2.88 | 0.05 | 0.07 | 2.81 | 3.06 | 100.0% |
| Grailsort (sqrt n) | 2.21 | 2.26 | 0.10 | 0.17 | 2.15 | 2.50 | 100.0% |
| Miausort v2 (in-place) | 3.07 | 3.11 | 0.10 | 0.13 | 3.00 | 3.41 | 100.0% |
| Miausort v2 (sqrt n) | 2.67 | 2.69 | 0.12 | 0.09 | 2.59 | 3.30 | 100.0% |
| Miausort v3 (in-place) | 1.83 | 1.84 | 0.05 | 0.06 | 1.78 | 2.05 | 100.0% |
| Miausort v3 (sqrt n) | 1.25 | 1.27 | 0.05 | 0.04 | 1.23 | 1.53 | 100.0% |
| Timsort | 1.26 | 1.27 | 0.03 | 0.03 | 1.23 | 1.41 | 100.0% |
| Introsort | 0.58 | 0.59 | 0.01 | 0.01 | 0.57 | 0.63 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 36.77 | 36.87 | 0.44 | 0.73 | 36.16 | 38.01 | 100.0% |
| Grailsort (sqrt n) | 25.76 | 25.78 | 0.15 | 0.16 | 25.26 | 26.13 | 100.0% |
| Miausort v2 (in-place) | 35.50 | 35.51 | 0.65 | 0.33 | 34.64 | 39.61 | 100.0% |
| Miausort v2 (sqrt n) | 28.42 | 28.47 | 0.29 | 0.20 | 28.15 | 29.76 | 100.0% |
| Miausort v3 (in-place) | 22.57 | 22.59 | 0.13 | 0.14 | 22.42 | 23.10 | 100.0% |
| Miausort v3 (sqrt n) | 15.35 | 15.36 | 0.07 | 0.10 | 15.25 | 15.57 | 100.0% |
| Timsort | 16.96 | 16.97 | 0.05 | 0.07 | 16.84 | 17.14 | 100.0% |
| Introsort | 7.89 | 7.92 | 0.12 | 0.10 | 7.77 | 8.41 | 100.0% |
## Mean of 50 runs (gen_few_unique)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.24 | 0.24 | 0.00 | 0.00 | 0.24 | 0.25 | 100.0% |
| Grailsort (sqrt n) | 0.22 | 0.22 | 0.01 | 0.00 | 0.22 | 0.27 | 100.0% |
| Miausort v2 (in-place) | 0.14 | 0.15 | 0.01 | 0.00 | 0.14 | 0.19 | 100.0% |
| Miausort v2 (sqrt n) | 0.14 | 0.14 | 0.00 | 0.00 | 0.14 | 0.16 | 100.0% |
| Miausort v3 (in-place) | 0.10 | 0.10 | 0.01 | 0.00 | 0.10 | 0.17 | 100.0% |
| Miausort v3 (sqrt n) | 0.10 | 0.10 | 0.00 | 0.00 | 0.10 | 0.10 | 100.0% |
| Timsort | 0.06 | 0.06 | 0.00 | 0.00 | 0.06 | 0.08 | 100.0% |
| Introsort | 0.03 | 0.03 | 0.00 | 0.00 | 0.02 | 0.03 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 3.35 | 3.36 | 0.03 | 0.03 | 3.33 | 3.43 | 100.0% |
| Grailsort (sqrt n) | 3.17 | 3.19 | 0.11 | 0.02 | 3.16 | 3.94 | 100.0% |
| Miausort v2 (in-place) | 2.33 | 2.35 | 0.04 | 0.01 | 2.32 | 2.50 | 100.0% |
| Miausort v2 (sqrt n) | 2.31 | 2.34 | 0.17 | 0.06 | 2.25 | 3.39 | 100.0% |
| Miausort v3 (in-place) | 2.20 | 2.20 | 0.02 | 0.01 | 2.18 | 2.26 | 100.0% |
| Miausort v3 (sqrt n) | 1.13 | 1.13 | 0.02 | 0.01 | 1.12 | 1.24 | 100.0% |
| Timsort | 1.10 | 1.10 | 0.02 | 0.01 | 1.09 | 1.19 | 100.0% |
| Introsort | 0.39 | 0.40 | 0.01 | 0.00 | 0.39 | 0.43 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 40.80 | 40.76 | 0.29 | 0.17 | 40.16 | 41.67 | 100.0% |
| Grailsort (sqrt n) | 38.90 | 38.83 | 0.53 | 0.74 | 37.98 | 40.48 | 100.0% |
| Miausort v2 (in-place) | 26.60 | 26.63 | 0.15 | 0.07 | 26.49 | 27.54 | 100.0% |
| Miausort v2 (sqrt n) | 26.87 | 26.78 | 0.31 | 0.23 | 26.18 | 27.73 | 100.0% |
| Miausort v3 (in-place) | 24.97 | 24.96 | 0.20 | 0.09 | 24.41 | 25.96 | 100.0% |
| Miausort v3 (sqrt n) | 13.28 | 13.29 | 0.10 | 0.05 | 13.21 | 13.96 | 100.0% |
| Timsort | 12.82 | 12.84 | 0.16 | 0.07 | 12.70 | 13.85 | 100.0% |
| Introsort | 5.49 | 5.50 | 0.03 | 0.03 | 5.44 | 5.59 | 100.0% |
## Mean of 50 runs (gen_sorted)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Grailsort (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.10 | 0.11 | 0.01 | 0.01 | 0.10 | 0.15 | 100.0% |
| Grailsort (sqrt n) | 0.07 | 0.07 | 0.00 | 0.00 | 0.07 | 0.08 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.02 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 1.65 | 1.66 | 0.04 | 0.04 | 1.62 | 1.87 | 100.0% |
| Grailsort (sqrt n) | 0.88 | 0.88 | 0.02 | 0.01 | 0.85 | 0.95 | 100.0% |
| Miausort v2 (in-place) | 0.04 | 0.05 | 0.00 | 0.00 | 0.04 | 0.05 | 100.0% |
| Miausort v2 (sqrt n) | 0.04 | 0.04 | 0.00 | 0.00 | 0.04 | 0.05 | 100.0% |
| Miausort v3 (in-place) | 0.04 | 0.05 | 0.00 | 0.00 | 0.04 | 0.05 | 100.0% |
| Miausort v3 (sqrt n) | 0.05 | 0.05 | 0.00 | 0.00 | 0.05 | 0.05 | 100.0% |
| Timsort | 0.05 | 0.05 | 0.00 | 0.00 | 0.04 | 0.05 | 100.0% |
| Introsort | 0.21 | 0.21 | 0.01 | 0.00 | 0.21 | 0.26 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 22.30 | 22.33 | 0.18 | 0.18 | 22.00 | 23.10 | 100.0% |
| Grailsort (sqrt n) | 10.28 | 10.29 | 0.12 | 0.17 | 10.06 | 10.70 | 100.0% |
| Miausort v2 (in-place) | 0.49 | 0.49 | 0.01 | 0.01 | 0.48 | 0.54 | 100.0% |
| Miausort v2 (sqrt n) | 0.47 | 0.47 | 0.01 | 0.01 | 0.46 | 0.50 | 100.0% |
| Miausort v3 (in-place) | 0.47 | 0.47 | 0.02 | 0.01 | 0.45 | 0.59 | 100.0% |
| Miausort v3 (sqrt n) | 0.48 | 0.49 | 0.01 | 0.01 | 0.47 | 0.55 | 100.0% |
| Timsort | 0.51 | 0.51 | 0.02 | 0.02 | 0.47 | 0.62 | 100.0% |
| Introsort | 3.36 | 3.37 | 0.05 | 0.06 | 3.28 | 3.56 | 100.0% |
## Mean of 50 runs (gen_reversed)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.16 | 0.16 | 0.01 | 0.00 | 0.16 | 0.20 | 100.0% |
| Grailsort (sqrt n) | 0.14 | 0.14 | 0.00 | 0.00 | 0.13 | 0.14 | 100.0% |
| Miausort v2 (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.02 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Timsort | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.02 | 100.0% |
| Introsort | 0.02 | 0.02 | 0.00 | 0.00 | 0.02 | 0.02 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 2.29 | 2.30 | 0.04 | 0.03 | 2.28 | 2.50 | 100.0% |
| Grailsort (sqrt n) | 1.63 | 1.63 | 0.01 | 0.02 | 1.62 | 1.68 | 100.0% |
| Miausort v2 (in-place) | 0.14 | 0.14 | 0.00 | 0.00 | 0.14 | 0.16 | 100.0% |
| Miausort v2 (sqrt n) | 0.14 | 0.14 | 0.00 | 0.00 | 0.14 | 0.14 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.14 | 0.00 | 0.00 | 0.14 | 0.16 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.14 | 0.00 | 0.00 | 0.14 | 0.15 | 100.0% |
| Timsort | 0.07 | 0.07 | 0.00 | 0.00 | 0.07 | 0.08 | 100.0% |
| Introsort | 0.24 | 0.24 | 0.01 | 0.01 | 0.23 | 0.28 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 28.42 | 28.44 | 0.29 | 0.17 | 27.89 | 30.03 | 100.0% |
| Grailsort (sqrt n) | 18.48 | 18.51 | 0.23 | 0.32 | 18.09 | 19.24 | 100.0% |
| Miausort v2 (in-place) | 1.47 | 1.48 | 0.03 | 0.03 | 1.42 | 1.56 | 100.0% |
| Miausort v2 (sqrt n) | 1.47 | 1.48 | 0.03 | 0.04 | 1.43 | 1.60 | 100.0% |
| Miausort v3 (in-place) | 1.46 | 1.47 | 0.05 | 0.05 | 1.41 | 1.64 | 100.0% |
| Miausort v3 (sqrt n) | 1.47 | 1.48 | 0.03 | 0.05 | 1.43 | 1.56 | 100.0% |
| Timsort | 0.76 | 0.76 | 0.02 | 0.02 | 0.73 | 0.80 | 100.0% |
| Introsort | 3.55 | 3.58 | 0.16 | 0.05 | 3.50 | 4.67 | 100.0% |
## Mean of 50 runs (gen_reversed_segments)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Grailsort (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Miausort v2 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v2 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (in-place) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |
| Miausort v3 (sqrt n) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Timsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 100.0% |
| Introsort | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 0.32 | 0.32 | 0.01 | 0.00 | 0.31 | 0.38 | 100.0% |
| Grailsort (sqrt n) | 0.29 | 0.29 | 0.00 | 0.00 | 0.28 | 0.30 | 100.0% |
| Miausort v2 (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.02 | 100.0% |
| Miausort v2 (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 100.0% |
| Miausort v3 (in-place) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.03 | 100.0% |
| Miausort v3 (sqrt n) | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.02 | 100.0% |
| Timsort | 0.07 | 0.07 | 0.00 | 0.00 | 0.07 | 0.09 | 100.0% |
| Introsort | 0.02 | 0.02 | 0.00 | 0.00 | 0.02 | 0.02 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 3.43 | 3.46 | 0.10 | 0.10 | 3.36 | 3.95 | 100.0% |
| Grailsort (sqrt n) | 2.88 | 2.89 | 0.09 | 0.13 | 2.73 | 3.10 | 100.0% |
| Miausort v2 (in-place) | 0.14 | 0.15 | 0.01 | 0.01 | 0.14 | 0.17 | 100.0% |
| Miausort v2 (sqrt n) | 0.14 | 0.15 | 0.01 | 0.01 | 0.14 | 0.16 | 100.0% |
| Miausort v3 (in-place) | 0.14 | 0.15 | 0.00 | 0.01 | 0.14 | 0.17 | 100.0% |
| Miausort v3 (sqrt n) | 0.14 | 0.15 | 0.00 | 0.00 | 0.14 | 0.16 | 100.0% |
| Timsort | 1.12 | 1.15 | 0.12 | 0.05 | 1.09 | 1.91 | 100.0% |
| Introsort | 0.24 | 0.25 | 0.01 | 0.01 | 0.24 | 0.28 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 42.18 | 42.52 | 1.08 | 2.05 | 41.30 | 45.42 | 100.0% |
| Grailsort (sqrt n) | 30.82 | 31.16 | 0.69 | 1.16 | 30.24 | 33.29 | 100.0% |
| Miausort v2 (in-place) | 1.54 | 1.55 | 0.04 | 0.05 | 1.50 | 1.65 | 100.0% |
| Miausort v2 (sqrt n) | 1.53 | 1.54 | 0.04 | 0.05 | 1.47 | 1.71 | 100.0% |
| Miausort v3 (in-place) | 1.53 | 1.53 | 0.03 | 0.04 | 1.48 | 1.67 | 100.0% |
| Miausort v3 (sqrt n) | 1.53 | 1.54 | 0.03 | 0.04 | 1.48 | 1.68 | 100.0% |
| Timsort | 12.75 | 12.77 | 0.18 | 0.23 | 12.48 | 13.27 | 100.0% |
| Introsort | 3.80 | 3.81 | 0.09 | 0.10 | 3.66 | 4.09 | 100.0% |
