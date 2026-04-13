# Miau Sort
An adaptive (run aware), stable, in-place O(n log n) worst-case sort taking ideas from [Grail Sort](https://github.com/Mrrl/GrailSort) among many others.
In development, but currently has the following features:
- Run scans with stable reversals
- Early exits in event of run monotony
- O(n) worst-case key collection (as of v3), thanks to bzy for O(n) run repair from dustsort (also means that this is now comparison optimal, doing `n log n + O(n)` comparisons in the worst-case) 
- O(sqrt n) block tag sorting (from [Holy Grail Sort](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file))
- Optimized key redistribution
- Rewritten merge as of v3 (from [Adaptive Grail Sort](https://github.com/Gaming32/ArrayV/blob/main/src/main/java/io/github/arrayv/sorts/hybrid/AdaptiveGrailSort.java))
- Block cycle sort when allocated memory, further reducing writes.
- Galloping with more optimized heuristics

Currently working on:
- Porting to more practical languages (Java, C, Rust)

Acknowledgements:
- Amari [(double-a git)](https://git.a-a.dev/amari) for ideas from Helium Sort, and for the development of [UniV](https://git.a-a.dev/amari/UniV), the visualizer I used for the majority of this algorithm's development.
- The Holy Grailsort team [(github)](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file) for improved key sorting, rotations and block selection sort
- aphitorite and Flanlaina for ideas from Adaptive Grail Sort
- bzy ([github](https://github.com/bzyjin), [codeberg](https://codeberg.org/bzy/))  for help with optimised key collection and buffer redistribution (ideas from [dustsort](https://github.com/bzyjin/dustsort/tree/main)) as well as helping with debugging

# Python benchmarks:
Now [here](https://github.com/miau-watermelon/miausort/blob/main/Python/benchmarks.md). Keep in mind that Python isn't the most reliable, even if I tried to mitigate this by taking medians and disabling the garbage collector.
