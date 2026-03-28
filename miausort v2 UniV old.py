class MiauSort:
    minRun = 16
    minGallop = 7
    lastBuffer = 0
    
    def swap(self, arr, a, b):
        arr[a], arr[b] = arr[b], arr[a]
    
    def reverse(self, arr, start, end):
        end -= 1
        while start < end:
            self.swap(arr, start, end)
            start += 1
            end -= 1
    
    def blockSwapForward(self, arr, a, b, n):
        for i in range(n):
            self.swap(arr, a+i, b+i)
    
    def blockSwapBackward(self, arr, a, b, n):
        a -= 1
        b -= 1
        for i in range(n):
            self.swap(arr, a-i, b-i)

    def insert(self, arr, src, dest):
        x = arr[src]
        if src < dest:
            for i in range(src, dest):
                arr[i] = arr[i+1]
        else:
            for i in range(src, dest, -1):
                arr[i] = arr[i-1]
        arr[dest] = x
    
    def blockSize(self, n, m):
        d = m*m - (n<<2)
        if d < 0:
            return 0
        b = int((m+d**0.5)/2)
        
        while b+n//b > m:
            b -= 1
        
        return b
    
    def sqrtPow2(self, n):
        b = 1
        while b*b < n:
            b *= 2
        return b
    
    def buildRunsBackward(self, arr, start, end):
        minRun = self.minRun
        runs = 0
        currEnd = end
        while currEnd > start:
            currStart = self.countRunBackward(arr, start, currEnd)
            runLen = currEnd-currStart
            if currStart == start:
                runs += 1
                break
            
            if runLen < minRun:
                ext = max(currEnd-minRun, start)
                self.expSortBackward(arr, ext, currEnd, currStart)
                runs += 1
                currEnd = ext
                continue
            
            currEnd = currStart+runLen%minRun
            runs += 1
            
            if runLen%minRun == 0:
                continue
            
            ext = max(currEnd-minRun, start)
            self.expSortBackward(arr, ext, currEnd, currStart)
            runs += 1
            currEnd = ext
        return runs
    
    def countRunBackward(self, arr, start, end):
        n = end - start
        if n < 1:
            return end

        i = 1
        while i < n and arr[end-i-1] <= arr[end-i]:
            i += 1

        if i >= n or (i > 1 and arr[end-1] > arr[end-i]):
            return end-i

        head = i
        l = i

        while True:
            i += 1
            while i < n and arr[end-i-1] <= arr[end-i]:
                i += 1
            if i > l+1 and arr[end-l-1] > arr[end-i]:
                p = self.expSearchBackward(arr, arr[end-l-1], end-i, end-l, False)
                self.reverse(arr, p, end-l)
                l = end-p
                break

            self.reverse(arr, end-i, end-l)
            l = i

            if i >= n:
                break

        if head < l:
            self.reverse(arr, end-head, end)
            self.reverse(arr, end-l, end)

        return end-l
    
    def expSearchForward(self, arr, target, start, end, LR):
        if (arr[start] > target if LR else arr[start] >= target):
            return start
        i = 1
        while start+i < end and (arr[start+i] <= target if LR else arr[start+i] < target):
            i <<= 1
        lo = start+(i>>1)
        hi = min(start+i, end)
        while lo < hi:
            m = (lo+hi)>>1
            if (arr[m] <= target if LR else arr[m] < target):
                lo = m+1
            else:
                hi = m
        return lo
    
    def expSearchBackward(self, arr, target, start, end, LR):
        end -= 1
        if (arr[end] <= target if LR else arr[end] < target):
            return end+1
        i = 1
        while end-i >= start and (arr[end-i] > target if LR else arr[end-i] >= target):
            i <<= 1
        lo = max(end-i, start)
        hi = end-(i>>1)
        while lo < hi:
            m = (lo+hi)>>1
            if (arr[m] > target if LR else arr[m] >= target):
                hi = m
            else:
                lo = m+1
        return lo
    
    def expSortBackward(self, arr, start, end, hint):
        spd = UniV_getSpeed()
        UniV_setSpeed(spd*3)
        for i in range(hint-1, start-1, -1):
            self.insert(arr, i, self.expSearchForward(arr, arr[i], i+1, end, False)-1)
        UniV_setSpeed(spd)
    
    def shellsort(self, arr, start, end): # Only used for sorting buffers since they contain fully distinct values
        gaps = [6148184740, 2769452586, 1427501165, 561937462, 253124983, 114020263, 51360479, 23135351, 10528127, 4697153, 2131981, 973657, 443557, 197803, 89129, 40354, 18118, 8129, 3659, 1636, 701, 301, 132, 57, 23, 10, 4, 1]

        n = end-start
        for gap in gaps:
            if gap > n:
                continue

            for i in range(start+gap, end):
                key = arr[i]
                j = i

                while j >= start+gap and arr[j-gap] > key:
                    arr[j] = arr[j-gap]
                    j -= gap

                arr[j] = key
    
    def tagSort(self, arr, tagStart, tagCount, midTag): # Just a stable partition - means tags can be sorted in O(sqrt n) :D
        bufferStart = self.bufferStart
        currTag = tagStart
        bufferSwaps = 0
        
        while currTag < tagStart+tagCount:
            if arr[currTag] < midTag:
                if bufferSwaps != 0:
                    self.swap(arr, currTag, currTag-bufferSwaps)
            else:
                self.swap(arr, currTag, bufferStart+bufferSwaps)
                bufferSwaps += 1
            currTag += 1
        self.blockSwapForward(arr, currTag-bufferSwaps, bufferStart, bufferSwaps)
    
    def smartCollect(self, arr, start, end, target): # Probably the most complex part of the algorithm :P
        n = end-start
        runSize = self.minRun
        keysStart = start
        keyCount = 0
        runStart = start
        runEnd = start+n%runSize # Handling of 'remainder' runs (elements that don't fit in a multiple of minRun)
        if runEnd == end:
            runEnd += runSize # First run is a full run if no remainder
        
        lastRotated = start
        while runStart < end and keyCount < target:
            currRun = (end-runEnd+runSize-1)//runSize # Important calculation
            j = runStart
            searchStart = keysStart
            while j < runEnd:
                val = arr[j]
                pos = self.expSearchForward(arr, val, searchStart, keysStart + keyCount, False) # See if the current value at j can be added to keys
                searchStart = pos # Update to not have to start search from start every time
                
                if pos == keysStart+keyCount or arr[pos] != val: # Value at j is distinct from keys - can insert
                    self.insert(arr, j, pos)
                    keyCount += 1
                    j += 1
                else:
                    breakOuter = False
                    while runStart < end and not breakOuter: # Try to find a value distinct from keys
                        while j < runEnd:
                            val = arr[j]
                            pos = self.expSearchForward(arr, val, searchStart, keysStart+keyCount, False)
                            searchStart = pos
                            if pos == keysStart+keyCount or arr[pos] != val: # Found - can break
                                breakOuter = True
                                break
                            j = self.expSearchForward(arr, arr[pos], j, runEnd, True)
                        
                        if breakOuter:
                            self.rotate(arr, keysStart, keysStart+keyCount, min(j,end))
                            ofs = j-(keysStart+keyCount)
                            searchStart += ofs
                            keysStart += ofs
                            lastRotated = min(j+runSize, end)
                            break
                        
                        if keyCount >= runSize and currRun % 2 == 0: # Alternatively, move keys to a position where remaining runs can be merged if there are enough
                            self.rotate(arr, keysStart, keysStart+keyCount, min(j,end))
                            ofs = j-(keysStart+keyCount)
                            searchStart += ofs
                            keysStart += ofs
                            lastRotated = min(j+runSize, end)
                            break
                        
                        if j < end:
                            searchStart = self.expSearchForward(arr, arr[j], keysStart, keysStart+keyCount, False) # Reset for next run and continue searching
                        
                        runStart = runEnd
                        runEnd = min(runStart+runSize, end)
                        while runEnd < end and arr[runEnd-1] <= arr[runEnd]: # Find end of the current run (most useful for long sorted sections)
                            runEnd = min(runEnd+runSize, end)
                        currRun = (end-runEnd+runSize-1)//runSize # Recalculate current run from end
                
                if keyCount >= target: # Enough found - stop searching
                    break
            
            if keyCount >= target: # Same as above
                break
            
            if keyCount >= runSize and currRun % 2 == 0: # Enough keys have been collected to increase merge size - both an optimisation and a way to guarantee O(n) worst-case
                self.bufferStart = keysStart
                for right in range(end, runEnd, -2*runSize):
                    mid = max(runEnd, right-runSize)
                    left = max(runEnd, right-2*runSize)
                    
                    if left == mid:
                        break
                    
                    skip, l, r = self.checkBounds(arr, left, mid, right)
                    if skip:
                        continue
                    self.mergeBuf(arr, l, mid, r)
                
                self.shellsort(arr, keysStart, keysStart+self.lastBuffer)
                self.lastBuffer = 0
                runSize *= 2
            
            runStart = runEnd
            runEnd += runSize
        
        if keyCount >= target: # Enough to use smart tag sorting, so preserve order
            self.rotate(arr, start, keysStart, keysStart+keyCount)
        else: # Otherwise, we don't care and scroll keys to start - Faster, but does not preserve order of keys
            dist = keysStart-start
            self.blockSwapBackward(arr, keysStart+keyCount, keysStart, dist)
        
        keysEnd = start+keyCount
        lastRotated += (end-lastRotated)%runSize # Snap to rightmost run end
        self.buildRunsBackward(arr, keysEnd, lastRotated)
        
        self.bufferStart = start
        mergeN = self.minRun
        while mergeN < runSize: # Fix previously touched runs
            right = lastRotated
            while right > keysEnd:
                mid = max(keysEnd, right-mergeN)
                left = max(keysEnd, right-2*mergeN)
                
                if left == mid:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right)
                if not skip:
                    self.mergeBuf(arr, l, mid, r)
                
                right -= 2*mergeN
            mergeN *= 2
        
        mergeN = runSize
        while mergeN <= keyCount: # Merge rest of array until no more can fit inside the internal buffer
            right = end
            while right > keysEnd:
                mid = max(keysEnd, right-mergeN)
                left = max(keysEnd, right-2*mergeN)
                
                if left == mid:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right)
                if not skip:
                    self.mergeBuf(arr, l, mid, r)
                
                right -= 2*mergeN
            mergeN *= 2
        
        if keyCount >= target:
            self.shellsort(arr, start, start+self.lastBuffer)
            self.lastBuffer = 0
        
        self.minRun = max(mergeN//2, runSize)
        return keyCount
    
    def mergeDecide(self, arr, start, mid, end): # Helper
        lenA = mid-start
        lenB = end-mid
        
        if min(lenA, lenB) <= self.buffer:
            self.mergeBuf(arr, start, mid, end)
        else:
            self.lazyStableMerge(arr, start, mid, end, False)
    
    def mergeBuf(self, arr, start, mid, end): # Ensures merge direction matches buffer's capacity
        if mid-start < end-mid:
            self.lastBuffer = max(self.lastBuffer, mid-start)
            self.mergeBufForward(arr, start, mid, end)
        else:
            self.lastBuffer = max(self.lastBuffer, end-mid)
            self.mergeBufBackward(arr, start, mid, end)
    
    def checkBounds(self, arr, start, mid, end): # Merge optimiser - skips where merges aren't needed and shrinks bounds
        if arr[mid-1] <= arr[mid]:
            return True, start, end
        
        start = self.expSearchForward(arr, arr[mid], start, mid, True)
        end = self.expSearchBackward(arr, arr[mid-1], mid, end, False)
        
        if arr[start] > arr[end-1]:
            self.rotate(arr, start, mid, end)
            return True, start, end
        
        return False, start, end

    def mergeBufForward(self, arr, start, mid, end): # The galloping merges are actually pretty useful for something you'd expect to only be working on sqrt n values...
        minGallop = self.minGallop
        countLeft = countRight = 0

        bufferStart = self.bufferStart
        leftLen = mid-start
        self.blockSwapForward(arr, start, bufferStart, leftLen) # Move smaller half into internal buffer

        i, j, dest = bufferStart, mid, start
        while i < bufferStart+leftLen and j < end:
            while i < bufferStart+leftLen and j < end and max(countLeft, countRight) < minGallop: # Normal merge phase
                if arr[i] <= arr[j]:
                    self.swap(arr, i, dest) # Using swaps to avoid deleting the buffer
                    countLeft += 1; countRight = 0
                    i += 1
                else:
                    self.swap(arr, j, dest)
                    countRight += 1; countLeft = 0
                    j += 1
                dest += 1

            while i < bufferStart+leftLen and j < end: # Galloping merge phase
                countLeft = self.expSearchForward(arr, arr[j], i, bufferStart+leftLen, True)-i
                if countLeft:
                    self.blockSwapForward(arr, i, dest, countLeft)
                    i += countLeft; dest += countLeft
                    if i >= bufferStart+leftLen:
                        break
                self.swap(arr, j, dest)
                j += 1; dest += 1
                if j >= end:
                    break
                countRight = self.expSearchForward(arr, arr[i], j, end, False)-j
                if countRight:
                    self.blockSwapForward(arr, j, dest, countRight)
                    j += countRight; dest += countRight
                    if j >= end:
                        break
                self.swap(arr, i, dest)
                i += 1; dest += 1
                if i >= bufferStart+leftLen:
                    break

                if max(countLeft, countRight) < minGallop:
                    minGallop += 2
                    break

        while i < bufferStart+leftLen:
            self.swap(arr, i, dest)
            i += 1
            dest += 1

        self.minGallop = minGallop
    
    def mergeBufBackward(self, arr, start, mid, end): # So you know what I said about the galloping merges? Not so much in this direction.
        minGallop = self.minGallop
        countLeft = countRight = 0

        bufferStart = self.bufferStart
        rightLen = end-mid
        self.blockSwapBackward(arr, bufferStart+rightLen, end, rightLen)

        i, j, dest = mid-1, bufferStart+rightLen-1, end-1
        while i >= start and j >= bufferStart:
            while i >= start and j >= bufferStart and max(countLeft, countRight) < minGallop: # Normal merge phase
                if arr[j] >= arr[i]:
                    self.swap(arr, j, dest)
                    countRight += 1; countLeft = 0
                    j -= 1
                else:
                    self.swap(arr, i, dest)
                    countLeft += 1; countRight = 0
                    i -= 1
                dest -= 1

            while i >= start and j >= bufferStart: # Galloping phase
                countLeft = i+1-self.expSearchBackward(arr, arr[j], start, i+1, True)
                if countLeft:
                    self.blockSwapBackward(arr, i+1, dest+1, countLeft) # It took me so long to figure out that the direction of the block swap was actually important...
                    i -= countLeft; dest -= countLeft
                    if i < start:
                        break

                self.swap(arr, j, dest)
                j -= 1; dest -= 1
                if j < bufferStart:
                    break

                countRight = j+1-self.expSearchBackward(arr, arr[i], bufferStart, j+1, False)
                if countRight:
                    self.blockSwapBackward(arr, j+1, dest+1, countRight)
                    j -= countRight; dest -= countRight
                    if j < bufferStart:
                        break

                self.swap(arr, i, dest)
                i -= 1; dest -= 1
                if i < start:
                    break

                if max(countLeft, countRight) < minGallop:
                    minGallop += 2 # Penalise when galloping is exited
                    break

        while j >= bufferStart: # Copy any remaining elements in buffer
            self.swap(arr, j, dest)
            j -= 1
            dest -= 1
        
        self.minGallop = minGallop # Globalise minGallop
    
    def lazyStableMerge(self, arr, start, mid, end, right): # Implementation from Helium Sort, optimised with exponential searches
        if mid-start < end-mid:
            s = start
            l = mid

            while s < l and l < end:
                if (arr[s] >= arr[l] if right else arr[s] > arr[l]):
                    p = self.expSearchForward(arr, arr[s], l, end, right)
                    self.rotate(arr, s, l, p)
                    s += p-l
                    l = p
                else:
                    s += 1
        else:
            s = end - 1
            l = mid - 1

            while s > l and l >= start:
                if (arr[l] >= arr[s]) if right else (arr[l] > arr[s]):
                    p = self.expSearchBackward(arr, arr[s], start, l, not right)
                    self.rotate(arr, p, l+1, s+1)
                    s -= l+1-p
                    l = p-1
                else:
                    s -= 1
    
    def blockMerge(self, arr, start, mid, end):
        tagsStart = self.tagsStart
        bufferStart = self.bufferStart
        blockLen = self.blockLen
        
        lenA, lenB = mid-start, end-mid
        
        blockCountA = lenA//blockLen
        blockCountB = lenB//blockLen
        
        blockCount = blockCountA+blockCountB
        
        remainderA = lenA%blockLen
        remainderB = lenB%blockLen
        
        blockStart = start+remainderA
        
        if not self.smartTagSort:
            self.shellsort(arr, tagsStart, tagsStart+blockCount)
        
        midTag = arr[tagsStart+blockCountA]
        
        self.blockSelectSort(arr, blockStart, blockCount, blockLen, blockCountA, midTag)
        self.cleanupBlocks(arr, start, end, blockCount, blockLen, remainderA, remainderB, midTag)
        
        if self.smartTagSort:
            self.tagSort(arr, tagsStart, blockCount, midTag)
    
    def blockSelectSort(self, arr, start, blockCount, blockLen, blockB, midTag):
        tagsStart = self.tagsStart
        endB = blockB+1 # Only look at blocks if the min is guaranteed to be here

        for i in range(blockCount):
            minIdx = i

            if arr[tagsStart+i] < midTag: # Determining value used to compare blocks based on which half they came from to preserve stability
                minVal = arr[start+i*blockLen] # First item of A block
            else:
                minVal = arr[start+(i+1)*blockLen-1] # Last item of B block

            for j in range(max(blockB,i+1), endB):
                if arr[tagsStart+j] < midTag:
                    blockVal = arr[start+j*blockLen]
                else:
                    blockVal = arr[start+(j+1)*blockLen-1]

                if blockVal < minVal or (blockVal == minVal and arr[tagsStart+j] < arr[tagsStart+minIdx]):
                    minVal = blockVal
                    minIdx = j

            if minIdx != i:
                self.blockSwapForward(arr, start+i*blockLen, start+minIdx*blockLen, blockLen)
                self.swap(arr, tagsStart+i, tagsStart+minIdx)

                if endB < blockCount and minIdx == endB-1: # Only increment if a swap has been made and the min was at the current end (only works on two sorted subarrays)
                    endB += 1
    
    def cleanupBlocks(self, arr, start, end, blockCount, blockLen, leftRemainder, rightRemainder, midTag):
        tagsStart = self.tagsStart

        mergeEnd = end
        blockEnd = end

        if rightRemainder > 0:
            rightStart = blockEnd-rightRemainder
            skip, l, r = self.checkBounds(arr, rightStart-blockLen, rightStart, mergeEnd)
            if not skip:
                self.mergeDecide(arr, l, rightStart, r)
            mergeEnd = r
            blockEnd -= rightRemainder
            last = blockCount-1
        else:
            blockEnd -= blockLen
            last = blockCount-2
        
        i = last
        while i >= 0: # Doing this backwards is the simplest way to do it stably, but is much less performant. Looking to replace this with something more efficient.
            blockStart = blockEnd-blockLen
            if arr[tagsStart+i] < midTag:
                skip, l, r = self.checkBounds(arr, blockStart, blockEnd, mergeEnd)
                if not skip:
                    self.mergeDecide(arr, l, blockEnd, r)
                mergeEnd = r
            blockEnd -= blockLen
            i -= 1

        if leftRemainder > 0:
            skip, l, r = self.checkBounds(arr, start, start+leftRemainder, mergeEnd)
            if not skip:
                self.mergeDecide(arr, l, start+leftRemainder, r)
    
    def rotate(self, arr, a, m, e): # Trinity rotation
        lenA = m-a
        lenB = e-m

        if lenA < 1 or lenB < 1:
            return

        if lenA == lenB: # Gries-Mills' best-case
            self.blockSwapForward(arr, a, m, lenA)
            return

        if lenA == 1: # Better to replace with auxiliary memory (only 8)
            self.insert(arr, a, e-1)
            return
        
        if lenB == 1:
            self.insert(arr, e-1, a)
            return

        b = m-1
        c = m
        d = e-1

        tmp = 0
        spd = UniV_getSpeed() # Sequential writes look slower in visualisers even if they're faster in the CPU. Speed up to compensate.
        UniV_setSpeed(spd*2)
        while a < b and c < d: # Cycle reversal rotation (very fast on average)
            tmp = arr[b]
            arr[b] = arr[a]
            b -= 1
            arr[a] = arr[c]
            a += 1
            arr[c] = arr[d]
            c += 1
            arr[d] = tmp
            d -= 1
        while a < b:
            tmp = arr[b]
            arr[b] = arr[a]
            b -= 1
            arr[a] = arr[d]
            a += 1
            arr[d] = tmp
            d -= 1
        while c < d:
            tmp = arr[c]
            arr[c] = arr[d]
            c += 1
            arr[d] = arr[a]
            d -= 1
            arr[a] = tmp
            a += 1
        UniV_setSpeed(spd)
        if a < d:
            self.reverse(arr, a, d+1)
    
    def sort(self, arr, start, end):
        n = end-start
        
        if n <= 2*self.minRun: # Run-aware exponential insertion sort
            hint = self.countRunBackward(arr, start, end)
            if hint != start:
                self.expSortBackward(arr, start, end, hint)
            return
        
        if self.buildRunsBackward(arr, start, end) < 2: # Only one run was found. No need to do anything else, so exit early.
            return
        
        self.blockLen = self.sqrtPow2(n)
        
        bufferTarget = self.blockLen
        tagTarget = n//bufferTarget
        
        target = bufferTarget+tagTarget
        keys = self.smartCollect(arr, start, end, target)
        if keys >= target:
            self.tagsStart = start
            self.tags = tagTarget
            self.bufferStart = self.tagsStart+self.tags
            self.buffer = bufferTarget
            recheck = False
            self.smartTagSort = True
        else:
            self.tagsStart = start
            self.blockLen = max(int(n**0.5), self.blockSize((end-start), keys))
            self.tags = (2*self.minRun)//self.blockLen
            self.bufferStart = self.tagsStart+self.tags
            self.buffer = keys-self.tags
            recheck = True
            self.smartTagSort = False
        regionStart = start+keys
        mergeN = self.minRun
        while mergeN < end-regionStart:
            if recheck:
                self.blockLen = max(int((2*mergeN)**0.5), self.blockSize((2*mergeN), keys))
                self.tags = (2*mergeN)//self.blockLen
                if self.tags > keys:
                    self.tags = keys
                    self.blockLen = (2*mergeN)//self.tags
                self.bufferStart = self.tagsStart+self.tags
                self.buffer = keys-self.tags
            
            for right in range(end, regionStart, -2*mergeN):
                mid = max(right-mergeN, regionStart)
                left = max(right-2*mergeN, regionStart)
                
                if mid == left:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right)
                
                if skip:
                    continue
                
                if r-mid > self.blockLen:
                    r += (right-r)%self.blockLen
                
                if min(mid-l, r-mid) <= self.blockLen:
                    self.mergeDecide(arr, l, mid, r)
                else:
                    self.blockMerge(arr, l, mid, r)
            mergeN *= 2
        
        if self.smartTagSort:
            self.shellsort(arr, self.bufferStart, self.bufferStart+self.lastBuffer)
        else:
            self.shellsort(arr, start, start+keys)
        
        self.redistBuffer(arr, start, start+keys, end)
    
    def redistBuffer(self, arr, start, mid, end): # Modified block swap merge - original implementation from https://sortingalgos.miraheze.org/wiki/Rotate_Merge_Sort#Block-Swap_Merge
        leftLen = mid-start
        rightLen = end-mid
        while min(leftLen, rightLen) > 16:
            leftLen = mid-start
            rightLen = end-mid
            
            if leftLen <= rightLen:
                if arr[start] > arr[mid+leftLen]:
                    pos = self.expSearchForward(arr, arr[start], mid+leftLen, end, False)
                    self.rotate(arr, start, mid, pos)
                    dist = pos-mid
                    start += dist+1
                    mid += dist
                    leftLen -= 1
            else:
                if arr[end-1] < arr[mid-rightLen]:
                    pos = self.expSearchBackward(arr, arr[end-1], start, mid-rightLen, True)
                    self.rotate(arr, pos, mid, end)
                    dist = mid-pos
                    end -= dist+1
                    mid -= dist
                    rightLen -= 1
            
            lo = 0
            hi = min(leftLen, rightLen)
            while lo < hi:
                m = (lo+hi)//2
                if arr[mid-1-m] > arr[mid+m]:
                    lo = m+1
                else:
                    hi = m
            d = lo
            
            self.blockSwapForward(arr, mid-d, mid, d)
            
            if leftLen <= rightLen:
                self.lazyStableMerge(arr, start, mid-d, mid, False)
                start = mid
                mid += d
            else:
                self.lazyStableMerge(arr, mid, mid+d, end, False)
                end = mid
                mid -= d
        
        self.lazyStableMerge(arr, start, mid, end, False)

@Sort("Block Merge Sorts", "Miausort", "Miausort")
def miauSortRun(arr):
    MiauSort().sort(arr, 0, len(arr))

@Rotation("Trinity", RotationMode.INDEXED)
def trinityRotate(arr, start, mid, end):
    MiauSort().rotate(arr, start, mid, end)
