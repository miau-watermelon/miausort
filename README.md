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
- Porting to more practical languages for use outside of the sorting algorithm visualiser I'm using (Java, C, Rust)
- Making key collection O(n) worst-case

Acknowledgements:
- Amari [(double-a git)](https://git.a-a.dev/amari) for ideas from Helium Sort and [UniV](https://git.a-a.dev/amari/UniV), the visualizer I used for the vast majority of this algorithm's development.
- The Holy Grailsort team [(github)](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file) for improved key sorting, rotations and block selection sort
- bzy ([github](https://github.com/bzyjin), [codeberg](https://codeberg.org/bzy/))  for help with optimised key collection (ideas from [dustsort](https://github.com/bzyjin/dustsort/tree/main)) and also help with debugging

# Python benchmarks:
(I tried to make it accurate, but keep in mind that Python isn't the most reliable)

## Mean of 50 runs (gen_random)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 3.80 | 3.91 | 0.41 | 0.20 | 3.70 | 6.30 | 100.0% |
| Grailsort (sqrt n) | 4.50 | 4.81 | 1.34 | 0.40 | 4.30 | 13.50 | 100.0% |
| Miausort v2 (in-place) | 4.90 | 5.31 | 1.89 | 0.30 | 4.60 | 17.30 | 100.0% |
| Miausort v2 (sqrt n) | 4.80 | 4.91 | 0.61 | 0.10 | 4.70 | 9.00 | 100.0% |
| Miausort v3 (in-place) | 3.55 | 3.95 | 2.35 | 0.10 | 3.40 | 20.20 | 100.0% |
| Miausort v3 (sqrt n) | 3.50 | 3.59 | 0.36 | 0.10 | 3.40 | 6.00 | 100.0% |
| Timsort | 4.00 | 5.62 | 8.90 | 0.22 | 3.70 | 66.30 | 100.0% |
| Introsort | 2.70 | 2.76 | 0.26 | 0.20 | 2.50 | 4.30 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 193.60 | 196.47 | 10.97 | 5.48 | 188.90 | 262.60 | 100.0% |
| Grailsort (sqrt n) | 176.40 | 176.74 | 8.60 | 2.30 | 164.60 | 216.10 | 100.0% |
| Miausort v2 (in-place) | 171.45 | 173.49 | 6.96 | 1.43 | 169.90 | 211.20 | 100.0% |
| Miausort v2 (sqrt n) | 204.10 | 205.82 | 6.41 | 3.55 | 200.30 | 241.60 | 100.0% |
| Miausort v3 (in-place) | 124.20 | 124.76 | 2.08 | 1.30 | 122.80 | 134.70 | 100.0% |
| Miausort v3 (sqrt n) | 124.70 | 125.15 | 1.97 | 1.40 | 123.40 | 135.40 | 100.0% |
| Timsort | 67.45 | 67.96 | 1.64 | 0.73 | 66.90 | 77.60 | 100.0% |
| Introsort | 33.30 | 33.38 | 0.47 | 0.23 | 33.00 | 36.40 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 2895.30 | 2904.24 | 46.64 | 58.37 | 2834.80 | 3030.20 | 100.0% |
| Grailsort (sqrt n) | 2239.00 | 2238.21 | 30.46 | 43.55 | 2182.90 | 2305.80 | 100.0% |
| Miausort v2 (in-place) | 3018.95 | 3033.25 | 78.39 | 88.60 | 2921.30 | 3318.90 | 100.0% |
| Miausort v2 (sqrt n) | 2589.60 | 2606.12 | 37.63 | 55.35 | 2561.80 | 2731.00 | 100.0% |
| Miausort v3 (in-place) | 1821.65 | 1837.67 | 39.25 | 52.48 | 1796.90 | 1966.10 | 100.0% |
| Miausort v3 (sqrt n) | 1245.55 | 1259.30 | 30.23 | 27.40 | 1233.70 | 1419.70 | 100.0% |
| Timsort | 1257.80 | 1266.46 | 25.36 | 25.97 | 1238.20 | 1354.60 | 100.0% |
| Introsort | 587.90 | 594.81 | 24.86 | 12.10 | 579.60 | 752.40 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 37442.50 | 37493.77 | 351.38 | 224.38 | 37110.70 | 39026.10 | 100.0% |
| Grailsort (sqrt n) | 26685.20 | 26691.13 | 112.69 | 146.60 | 26446.30 | 26968.90 | 100.0% |
| Miausort v2 (in-place) | 35312.90 | 35306.35 | 333.66 | 280.40 | 34408.40 | 36238.50 | 100.0% |
| Miausort v2 (sqrt n) | 28374.30 | 28379.53 | 217.97 | 191.75 | 27877.20 | 29431.90 | 100.0% |
| Miausort v3 (in-place) | 22642.55 | 22671.64 | 166.54 | 115.45 | 22463.80 | 23691.30 | 100.0% |
| Miausort v3 (sqrt n) | 15915.30 | 15907.52 | 133.63 | 135.85 | 15637.70 | 16436.70 | 100.0% |
| Timsort | 16882.20 | 16891.51 | 75.22 | 90.22 | 16711.00 | 17093.20 | 100.0% |
| Introsort | 7256.60 | 7345.31 | 239.93 | 99.65 | 7184.00 | 8636.10 | 100.0% |
## Mean of 50 runs (gen_few_unique)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 3.30 | 3.36 | 0.40 | 0.10 | 3.20 | 5.60 | 100.0% |
| Grailsort (sqrt n) | 3.90 | 3.96 | 0.37 | 0.12 | 3.70 | 6.00 | 100.0% |
| Miausort v2 (in-place) | 4.40 | 4.54 | 0.69 | 0.20 | 4.30 | 9.20 | 100.0% |
| Miausort v2 (sqrt n) | 5.00 | 5.14 | 0.64 | 0.30 | 4.70 | 9.20 | 100.0% |
| Miausort v3 (in-place) | 3.10 | 3.16 | 0.27 | 0.03 | 3.00 | 4.90 | 100.0% |
| Miausort v3 (sqrt n) | 3.20 | 3.26 | 0.45 | 0.10 | 3.00 | 6.20 | 100.0% |
| Timsort | 3.70 | 3.76 | 0.46 | 0.10 | 3.50 | 6.70 | 100.0% |
| Introsort | 2.00 | 2.06 | 0.16 | 0.10 | 1.90 | 3.10 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 232.65 | 237.70 | 16.71 | 2.13 | 230.00 | 327.00 | 100.0% |
| Grailsort (sqrt n) | 217.10 | 217.44 | 5.29 | 5.02 | 211.40 | 241.70 | 100.0% |
| Miausort v2 (in-place) | 151.40 | 152.36 | 3.61 | 1.00 | 150.10 | 175.50 | 100.0% |
| Miausort v2 (sqrt n) | 158.90 | 176.14 | 71.60 | 1.28 | 154.60 | 634.10 | 100.0% |
| Miausort v3 (in-place) | 102.35 | 102.78 | 2.26 | 0.93 | 100.60 | 115.40 | 100.0% |
| Miausort v3 (sqrt n) | 103.50 | 105.49 | 9.79 | 2.18 | 101.60 | 172.20 | 100.0% |
| Timsort | 66.45 | 66.69 | 1.31 | 0.70 | 65.40 | 74.70 | 100.0% |
| Introsort | 27.00 | 27.10 | 0.50 | 0.10 | 26.80 | 30.50 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 3297.95 | 3301.61 | 18.79 | 18.95 | 3277.40 | 3364.70 | 100.0% |
| Grailsort (sqrt n) | 3129.95 | 3149.93 | 111.12 | 30.73 | 3038.00 | 3823.10 | 100.0% |
| Miausort v2 (in-place) | 2349.35 | 2360.52 | 104.76 | 28.20 | 2279.20 | 3033.20 | 100.0% |
| Miausort v2 (sqrt n) | 2358.55 | 2388.06 | 106.02 | 23.15 | 2338.30 | 2983.70 | 100.0% |
| Miausort v3 (in-place) | 2196.85 | 2202.27 | 17.44 | 20.58 | 2180.00 | 2264.30 | 100.0% |
| Miausort v3 (sqrt n) | 1113.60 | 1115.45 | 11.58 | 11.33 | 1099.10 | 1162.10 | 100.0% |
| Timsort | 1080.35 | 1085.33 | 12.89 | 11.65 | 1073.10 | 1136.90 | 100.0% |
| Introsort | 396.80 | 397.79 | 6.34 | 3.15 | 391.70 | 437.30 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 41136.35 | 41197.20 | 192.51 | 205.10 | 40867.60 | 41881.80 | 100.0% |
| Grailsort (sqrt n) | 38729.95 | 38852.67 | 339.32 | 499.50 | 38465.20 | 40388.10 | 100.0% |
| Miausort v2 (in-place) | 27444.65 | 27439.32 | 90.57 | 120.12 | 27293.40 | 27678.60 | 100.0% |
| Miausort v2 (sqrt n) | 27624.50 | 27637.55 | 118.10 | 153.52 | 27464.30 | 28115.50 | 100.0% |
| Miausort v3 (in-place) | 25181.40 | 25182.91 | 201.12 | 126.05 | 24735.10 | 25791.00 | 100.0% |
| Miausort v3 (sqrt n) | 13825.60 | 13848.46 | 160.97 | 46.57 | 13709.50 | 14932.90 | 100.0% |
| Timsort | 12936.20 | 12971.46 | 164.92 | 64.62 | 12829.00 | 14040.00 | 100.0% |
| Introsort | 5470.30 | 5476.03 | 44.14 | 62.55 | 5412.00 | 5610.70 | 100.0% |
## Mean of 50 runs (gen_sorted)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 1.00 | 1.06 | 0.30 | 0.00 | 0.90 | 3.00 | 100.0% |
| Grailsort (sqrt n) | 1.50 | 1.58 | 0.29 | 0.10 | 1.50 | 3.60 | 100.0% |
| Miausort v2 (in-place) | 0.90 | 0.94 | 0.27 | 0.10 | 0.80 | 2.20 | 100.0% |
| Miausort v2 (sqrt n) | 1.10 | 1.11 | 0.21 | 0.00 | 1.00 | 2.50 | 100.0% |
| Miausort v3 (in-place) | 0.80 | 0.78 | 0.20 | 0.10 | 0.60 | 2.10 | 100.0% |
| Miausort v3 (sqrt n) | 0.80 | 0.81 | 0.23 | 0.10 | 0.70 | 2.30 | 100.0% |
| Timsort | 1.10 | 1.19 | 0.25 | 0.10 | 1.10 | 2.80 | 100.0% |
| Introsort | 1.00 | 1.02 | 0.21 | 0.00 | 0.90 | 2.40 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 101.75 | 104.49 | 10.62 | 3.10 | 100.50 | 176.00 | 100.0% |
| Grailsort (sqrt n) | 66.40 | 66.93 | 2.12 | 0.85 | 65.50 | 79.90 | 100.0% |
| Miausort v2 (in-place) | 3.90 | 3.99 | 0.28 | 0.10 | 3.80 | 5.80 | 100.0% |
| Miausort v2 (sqrt n) | 4.20 | 4.26 | 0.18 | 0.10 | 4.10 | 5.40 | 100.0% |
| Miausort v3 (in-place) | 3.80 | 4.08 | 0.80 | 0.10 | 3.70 | 8.60 | 100.0% |
| Miausort v3 (sqrt n) | 3.90 | 3.91 | 0.23 | 0.12 | 3.70 | 5.30 | 100.0% |
| Timsort | 4.50 | 4.58 | 0.41 | 0.03 | 4.40 | 7.20 | 100.0% |
| Introsort | 12.80 | 12.83 | 0.40 | 0.10 | 12.60 | 15.50 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 1677.75 | 1678.32 | 28.09 | 41.45 | 1635.40 | 1768.40 | 100.0% |
| Grailsort (sqrt n) | 892.55 | 889.62 | 12.39 | 19.53 | 862.20 | 920.90 | 100.0% |
| Miausort v2 (in-place) | 44.85 | 45.18 | 1.16 | 0.50 | 43.90 | 49.60 | 100.0% |
| Miausort v2 (sqrt n) | 44.70 | 46.07 | 7.28 | 2.70 | 42.90 | 95.40 | 100.0% |
| Miausort v3 (in-place) | 45.50 | 45.78 | 1.15 | 0.83 | 43.80 | 50.80 | 100.0% |
| Miausort v3 (sqrt n) | 44.85 | 45.03 | 1.37 | 1.60 | 43.10 | 50.40 | 100.0% |
| Timsort | 46.50 | 47.33 | 1.95 | 1.25 | 45.10 | 54.30 | 100.0% |
| Introsort | 214.25 | 219.27 | 28.27 | 3.42 | 210.60 | 410.90 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 22693.45 | 22734.16 | 239.57 | 138.75 | 22493.10 | 23884.10 | 100.0% |
| Grailsort (sqrt n) | 10460.55 | 10476.77 | 71.54 | 101.03 | 10362.90 | 10688.60 | 100.0% |
| Miausort v2 (in-place) | 483.50 | 486.17 | 11.32 | 10.73 | 467.90 | 527.50 | 100.0% |
| Miausort v2 (sqrt n) | 465.20 | 466.79 | 13.17 | 17.40 | 441.90 | 504.60 | 100.0% |
| Miausort v3 (in-place) | 466.05 | 467.19 | 14.65 | 19.45 | 439.90 | 516.20 | 100.0% |
| Miausort v3 (sqrt n) | 457.65 | 461.07 | 12.79 | 12.42 | 442.10 | 516.40 | 100.0% |
| Timsort | 481.50 | 483.24 | 10.81 | 8.00 | 466.30 | 538.30 | 100.0% |
| Introsort | 3306.35 | 3334.24 | 110.85 | 58.48 | 3220.00 | 3888.10 | 100.0% |
## Mean of 50 runs (gen_reversed)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 5.40 | 5.82 | 1.35 | 0.13 | 5.30 | 11.60 | 100.0% |
| Grailsort (sqrt n) | 6.10 | 6.31 | 1.25 | 0.10 | 5.90 | 14.60 | 100.0% |
| Miausort v2 (in-place) | 1.90 | 1.94 | 0.33 | 0.00 | 1.80 | 4.20 | 100.0% |
| Miausort v2 (sqrt n) | 2.10 | 2.14 | 0.33 | 0.00 | 2.00 | 4.40 | 100.0% |
| Miausort v3 (in-place) | 1.80 | 2.35 | 1.20 | 0.10 | 1.70 | 5.30 | 100.0% |
| Miausort v3 (sqrt n) | 1.80 | 1.89 | 0.33 | 0.10 | 1.70 | 4.10 | 100.0% |
| Timsort | 1.30 | 1.39 | 0.29 | 0.10 | 1.20 | 3.30 | 100.0% |
| Introsort | 2.80 | 3.48 | 4.07 | 0.10 | 2.70 | 31.90 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 162.70 | 165.20 | 5.63 | 3.62 | 161.60 | 189.60 | 100.0% |
| Grailsort (sqrt n) | 135.10 | 135.32 | 3.00 | 1.65 | 131.30 | 148.50 | 100.0% |
| Miausort v2 (in-place) | 12.60 | 12.80 | 0.74 | 0.10 | 12.40 | 17.20 | 100.0% |
| Miausort v2 (sqrt n) | 13.20 | 13.20 | 0.33 | 0.30 | 12.80 | 15.20 | 100.0% |
| Miausort v3 (in-place) | 12.70 | 12.88 | 0.72 | 0.10 | 12.60 | 17.40 | 100.0% |
| Miausort v3 (sqrt n) | 13.20 | 13.60 | 0.90 | 1.40 | 12.70 | 17.00 | 100.0% |
| Timsort | 6.70 | 6.98 | 0.79 | 0.20 | 6.50 | 11.60 | 100.0% |
| Introsort | 14.80 | 14.99 | 0.71 | 0.22 | 14.60 | 19.30 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 2310.60 | 2315.72 | 27.66 | 27.05 | 2276.80 | 2432.50 | 100.0% |
| Grailsort (sqrt n) | 1655.95 | 1684.37 | 166.98 | 18.48 | 1635.50 | 2847.10 | 100.0% |
| Miausort v2 (in-place) | 139.60 | 140.65 | 3.49 | 1.20 | 138.10 | 157.10 | 100.0% |
| Miausort v2 (sqrt n) | 144.20 | 144.71 | 2.26 | 3.27 | 141.00 | 150.00 | 100.0% |
| Miausort v3 (in-place) | 139.50 | 140.66 | 4.44 | 2.55 | 136.80 | 161.60 | 100.0% |
| Miausort v3 (sqrt n) | 142.10 | 141.63 | 1.94 | 3.35 | 138.30 | 146.00 | 100.0% |
| Timsort | 73.50 | 74.79 | 4.50 | 1.50 | 71.60 | 97.00 | 100.0% |
| Introsort | 239.40 | 240.10 | 4.77 | 3.85 | 235.90 | 268.40 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 28592.55 | 28589.83 | 161.04 | 152.55 | 28263.80 | 29141.90 | 100.0% |
| Grailsort (sqrt n) | 18429.55 | 18453.34 | 147.01 | 164.05 | 18034.40 | 18914.80 | 100.0% |
| Miausort v2 (in-place) | 1503.30 | 1500.82 | 23.01 | 26.60 | 1453.60 | 1558.00 | 100.0% |
| Miausort v2 (sqrt n) | 1541.35 | 1558.18 | 40.96 | 47.15 | 1510.20 | 1681.80 | 100.0% |
| Miausort v3 (in-place) | 1486.80 | 1486.96 | 32.99 | 38.50 | 1426.90 | 1627.60 | 100.0% |
| Miausort v3 (sqrt n) | 1505.00 | 1507.54 | 25.63 | 44.93 | 1466.10 | 1557.20 | 100.0% |
| Timsort | 796.95 | 802.03 | 28.10 | 40.18 | 756.70 | 876.80 | 100.0% |
| Introsort | 3647.95 | 3652.55 | 61.43 | 57.32 | 3566.60 | 3895.90 | 100.0% |
## Mean of 50 runs (gen_reversed_segments)

### Array length 10

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 5.10 | 5.14 | 0.31 | 0.12 | 5.00 | 7.20 | 100.0% |
| Grailsort (sqrt n) | 5.70 | 5.93 | 1.14 | 0.10 | 5.60 | 13.70 | 100.0% |
| Miausort v2 (in-place) | 1.80 | 1.85 | 0.35 | 0.00 | 1.70 | 4.20 | 100.0% |
| Miausort v2 (sqrt n) | 2.10 | 2.16 | 0.33 | 0.00 | 2.00 | 4.30 | 100.0% |
| Miausort v3 (in-place) | 1.90 | 1.89 | 0.21 | 0.10 | 1.80 | 3.30 | 100.0% |
| Miausort v3 (sqrt n) | 1.90 | 1.93 | 0.25 | 0.00 | 1.80 | 3.60 | 100.0% |
| Timsort | 3.80 | 3.93 | 0.46 | 0.10 | 3.70 | 6.90 | 100.0% |
| Introsort | 2.80 | 2.94 | 0.68 | 0.10 | 2.60 | 7.50 | 100.0% |

### Array length 100

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 314.10 | 316.35 | 8.49 | 2.65 | 311.80 | 360.00 | 100.0% |
| Grailsort (sqrt n) | 284.40 | 297.67 | 51.96 | 2.75 | 282.50 | 554.00 | 100.0% |
| Miausort v2 (in-place) | 12.80 | 13.00 | 0.96 | 0.20 | 12.60 | 18.90 | 100.0% |
| Miausort v2 (sqrt n) | 13.20 | 13.27 | 0.40 | 0.20 | 13.00 | 15.80 | 100.0% |
| Miausort v3 (in-place) | 16.60 | 16.81 | 0.98 | 0.87 | 15.60 | 20.50 | 100.0% |
| Miausort v3 (sqrt n) | 13.20 | 13.41 | 0.85 | 0.30 | 13.00 | 18.50 | 100.0% |
| Timsort | 67.05 | 68.17 | 4.23 | 1.17 | 66.10 | 94.70 | 100.0% |
| Introsort | 15.40 | 15.57 | 0.80 | 0.23 | 15.20 | 20.40 | 100.0% |

### Array length 1000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 3450.45 | 3448.36 | 34.81 | 50.07 | 3376.00 | 3545.40 | 100.0% |
| Grailsort (sqrt n) | 2790.30 | 2799.99 | 49.53 | 39.25 | 2748.40 | 3090.70 | 100.0% |
| Miausort v2 (in-place) | 143.10 | 142.67 | 2.46 | 4.45 | 138.30 | 147.60 | 100.0% |
| Miausort v2 (sqrt n) | 144.70 | 146.15 | 7.61 | 2.45 | 141.90 | 197.60 | 100.0% |
| Miausort v3 (in-place) | 143.00 | 143.18 | 2.66 | 2.90 | 137.90 | 155.20 | 100.0% |
| Miausort v3 (sqrt n) | 143.15 | 144.70 | 5.32 | 3.43 | 140.80 | 177.50 | 100.0% |
| Timsort | 1101.00 | 1100.64 | 23.85 | 17.57 | 1060.60 | 1224.80 | 100.0% |
| Introsort | 242.05 | 241.89 | 3.96 | 5.28 | 235.00 | 257.90 | 100.0% |

### Array length 10000

| Algorithm | Median (ms) | Mean (ms) | σ | IQR | Min | Max | Correct % |
|-----------|------------|-----------|---|-----|-----|-----|-----------|
| Grailsort (in-place) | 41640.10 | 41700.04 | 347.46 | 384.88 | 40975.40 | 42667.70 | 100.0% |
| Grailsort (sqrt n) | 31159.80 | 31258.89 | 342.52 | 361.15 | 30857.50 | 32821.80 | 100.0% |
| Miausort v2 (in-place) | 1524.90 | 1528.38 | 23.44 | 26.08 | 1492.30 | 1618.70 | 100.0% |
| Miausort v2 (sqrt n) | 1539.95 | 1542.98 | 18.15 | 15.80 | 1509.20 | 1620.80 | 100.0% |
| Miausort v3 (in-place) | 1520.50 | 1526.02 | 33.11 | 29.77 | 1479.40 | 1637.60 | 100.0% |
| Miausort v3 (sqrt n) | 1512.75 | 1527.43 | 87.53 | 46.60 | 1464.60 | 1941.10 | 100.0% |
| Timsort | 12545.55 | 12593.43 | 212.19 | 410.40 | 12131.90 | 12966.20 | 100.0% |
| Introsort | 3698.30 | 3699.56 | 49.40 | 57.45 | 3601.00 | 3834.00 | 100.0% |
