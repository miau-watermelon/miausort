A stable, in-place O(n log n) worst-case adaptive sort taking ideas from Grail Sort (https://github.com/Mrrl/GrailSort).
In development, but currently has these features:

- Run scans with stable reversals
- Early exits in event of run monotony
- O(n log n) worst-case key collection (unfortunately it isn't O(n) due to my lazy implementation of run fixing, studying alternative methods of this to push it down to O(n) :3)
- O(sqrt n) block tag sorting, improvement on Grail's O(n) sorting
- Significantly optimised key redistribution
- We don't talk about the actual merging (it's stable, in-place and O(n) worst-case, but tends to do around 1.25-1.5x as many writes as Grail. it is much simpler, though.)

Currently working on:
- Optimising merge
- Porting to more practical languages for use outside of the sorting algorithm visualiser I'm using (Normal Python, Java, C)
