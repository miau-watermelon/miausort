A stable, in-place O(n log n) worst-case adaptive sort taking ideas from [Grail Sort](https://github.com/Mrrl/GrailSort).
In development, but currently has the following features:
- Run scans with stable reversals
- Early exits in event of run monotony
- O(n log n) worst-case key collection (unfortunately it isn't O(n) due to my lazy implementation of run fixing, studying alternative methods of this to push it down to O(n) :3)
- O(sqrt n) block tag sorting, improvement on Grail's O(n) for this
- Significantly optimised key redistribution
- Rewritten merge as of v3 (from [Adaptive Grail Sort](https://github.com/Gaming32/ArrayV/blob/main/src/main/java/io/github/arrayv/sorts/hybrid/AdaptiveGrailSort.java))

Currently working on:
- Porting to more practical languages for use outside of the sorting algorithm visualiser I'm using (Normal Python, Java, C)
- Making key collection O(n) worst-case

Acknowledgements:
- Amari [(double-a git)](https://git.a-a.dev/amari) for ideas from Helium Sort and [UniV](https://git.a-a.dev/amari/UniV), the visualizer I used for the vast majority of this algorithm's development.
- The Holy Grailsort team [(github)](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file) for improved key sorting, rotations and block selection sort
- bzy ([github](https://github.com/bzyjin), [codeberg](https://codeberg.org/bzy/))  for help with optimised key collection (ideas from [dustsort](https://github.com/bzyjin/dustsort/tree/main)) and also help with debugging
