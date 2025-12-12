A stable, in-place O(n log n) worst-case adaptive sort taking ideas from Grail Sort (https://github.com/Mrrl/GrailSort).
In development, but currently has these features:
- Run scans with stable reversals
- Early exits in event of run monotony
- O(n log n) worst-case key collection (unfortunately it isn't O(n) due to my lazy implementation of run fixing, studying alternative methods of this to push it down to O(n) :3)
- O(sqrt n) block tag sorting, improvement on Grail's O(n) for this
- Significantly optimised key redistribution
- I optimised the merge (mechanics from Helium Sort)

Currently working on:
- Porting to more practical languages for use outside of the sorting algorithm visualiser I'm using (Normal Python, Java, C)
- Making key collection O(n) worst-case

Acknowledgements:
- Amari [(double-a git)](https://git.a-a.dev/amari) for ideas from Helium Sort, developing [UniV](https://git.a-a.dev/amari/UniV) (the visualiser I used for the vast majority of this algorithm's development) and emotional support when making this pushed my autistic ahh to tears (if you're reading this, i didn't say anything but i definitely did cry over this and talking helped so thanks ^w^)
- The Holy Grailsort team [(github)](https://github.com/HolyGrailSortProject/Holy-Grailsort?tab=readme-ov-file) for improved key sorting, rotations and block selection sort
- arctic [(github)](https://github.com/bzyjin) for help with optimised key collection (ideas from [dustsort](https://github.com/bzyjin/dustsort/tree/main)) and also helping with debugging sometimes :P
