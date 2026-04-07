# Miau Sort
A stable, in-place O(n log n) worst-case adaptive sort taking ideas from [Grail Sort](https://github.com/Mrrl/GrailSort).
In development, but currently has the following features:
- Run scans with stable reversals
- Early exits in event of run monotony
- O(n) worst-case key collection (as of v3), thanks to bzy for O(n) run repair from dustsort (also means that this is now comparison optimal, doing `n log n + O(n)` comparisons in the worst-case) 
- O(sqrt n) block tag sorting, improvement on Grail's O(n) for this
- Significantly optimised key redistribution
- Rewritten merge as of v3 (from [Adaptive Grail Sort](https://github.com/Gaming32/ArrayV/blob/main/src/main/java/io/github/arrayv/sorts/hybrid/AdaptiveGrailSort.java))
- Block cycle sort when allocated memory, further reducing writes.

Currently working on:
- Porting to more practical languages (Java, C, Rust)

Acknowledgements:
- Amari [(double-a git)](https://git.a-a.dev/amari) for ideas from Helium Sort and [UniV](https://git.a-a.dev/amari/UniV), the visualizer I used for the vast majority of this algorithm's development.
- The Holy Grailsort team [(github)](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file) for improved key sorting, rotations and block selection sort
- bzy ([github](https://github.com/bzyjin), [codeberg](https://codeberg.org/bzy/))  for help with optimised key collection (ideas from [dustsort](https://github.com/bzyjin/dustsort/tree/main)) and also help with debugging

# Python benchmarks:
Now [here](https://github.com/miau-watermelon/miausort/blob/main/Python/benchmarks.md).
