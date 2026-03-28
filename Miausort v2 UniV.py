"""
MIT License

Copyright (c) 2025 miau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class MiauSortOld:
    def __init__(self, mem):
        self.sizeSmall = 16
        self.minRun = self.sizeSmall
        self.minGallop = 7
        self.lastBuffer = 0
        self.rotateMaxAux = 8
        self.mem = mem
        self.buffer = None
    
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
    
    def arrayCopyForward(self, src, a, dest, b, n):
        for i in range(n):
            dest[b+i] = src[a+i]
    
    def arrayCopyBackward(self, src, a, dest, b, n):
        a -= 1
        b -= 1
        for i in range(n):
            dest[b-i] = src[a-i]
    
    def insertLeft(self, arr, src, dest):
        spd = UniV_getSpeed()
        UniV_setSpeed(spd*2)
        tmp = arr[src]
        for i in range(src, dest, -1):
            arr[i] = arr[i-1]
        arr[dest] = tmp
        UniV_setSpeed(spd)
    
    def insertRight(self, arr, src, dest):
        spd = UniV_getSpeed()
        UniV_setSpeed(spd*2)
        tmp = arr[src]
        for i in range(src, dest):
            arr[i] = arr[i+1]
        arr[dest] = tmp
        UniV_setSpeed(spd)
    
    def sqrtPow2(self, n):
        b = self.sizeSmall
        while b*b < n:
            b *= 2
        return b
    
    def floorPow2(self, n):
        n |= n>>1
        n |= n>>2
        n |= n>>4
        n |= n>>8
        n |= n>>16
        n |= n>>32
        return n-(n>>1)
    
    def countRun(self, arr, start, end):
        i = start + 1
        while i < end and arr[i] >= arr[i-1]: # Find non-descending run
            i += 1
        
        if i >= end or (i > start + 1 and arr[start] < arr[i-1]): # If ascending, no further work needed
            return i
        
        prev = arr[start]
        self.reverse(arr, start, i) # Reverse first segment
        
        while True:
            seg = i
            if seg >= end or arr[seg] >= prev: # Check if next segment is valid
                break
            
            curr = arr[seg]
            i += 1
            while i < end and arr[i] == curr:
                i += 1
                
            self.reverse(arr, seg, i) # Reverse the equal segment
            prev = curr
        
        self.reverse(arr, start, i) # Reverse entire sequence
        return i
    
    def buildRuns(self, arr, start, end):
        runs = 0
        minRun = self.minRun
        while start < end:
            currEnd = self.countRun(arr, start, end)
            runLen = currEnd-start
            
            if currEnd == end:
                runs += 1
                break
            
            if runLen < minRun:
                ext = min(start+minRun, end)
                self.expSort(arr, start, ext, currEnd)
                start = ext
                runs += 1
                continue
            
            start = currEnd - runLen%minRun
            runs += 1
            
            if start == currEnd:
                continue
            
            ext = min(start+minRun, end)
            self.expSort(arr, start, ext, currEnd)
            start = ext
            runs += 1
        return runs
    
    def expSearchForward(self, arr, target, start, end, right):
        if (arr[start] > target if right else arr[start] >= target):
            return start
        i = 1
        while start+i < end and (arr[start+i] <= target if right else arr[start+i] < target):
            i *= 2
        lo = start+i//2
        hi = min(start+i, end)
        while lo < hi:
            m = (lo+hi)//2
            if (arr[m] <= target if right else arr[m] < target):
                lo = m+1
            else:
                hi = m
        return lo
    
    def expSearchBackward(self, arr, target, start, end, right):
        end -= 1
        if (arr[end] <= target if right else arr[end] < target):
            return end+1
        i = 1
        while end-i >= start and (arr[end-i] > target if right else arr[end-i] >= target):
            i *= 2
        lo = max(end-i, start)
        hi = end-i//2
        while lo < hi:
            m = (lo+hi)//2
            if (arr[m] > target if right else arr[m] >= target):
                hi = m
            else:
                lo = m+1
        return lo
    
    def expSort(self, arr, start, end, hint):
        spd = UniV_getSpeed()
        UniV_setSpeed(spd*2)
        for i in range(hint, end):
            self.insertLeft(arr, i, self.expSearchBackward(arr, arr[i], start, i, True))
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
        if self.buffer is not None and len(self.buffer) >= tagCount:
            buf = self.buffer
            currTag = tagStart
            bufWrites = 0
            
            while currTag < tagStart+tagCount:
                if arr[currTag] < midTag:
                    if bufWrites != 0:
                        arr[currTag-bufWrites] = arr[currTag]
                else:
                    buf[bufWrites] = arr[currTag]
                    bufWrites += 1
                currTag += 1
            self.arrayCopyForward(buf, 0, arr, currTag-bufWrites, bufWrites)
        else:
            bufferStart = self.bufferStart
            currTag = tagStart
            bufSwaps = 0
            
            while currTag < tagStart+tagCount:
                if arr[currTag] < midTag:
                    if bufSwaps != 0:
                        self.swap(arr, currTag, currTag-bufSwaps)
                else:
                    self.swap(arr, currTag, bufferStart+bufSwaps)
                    bufSwaps += 1
                currTag += 1
            self.blockSwapForward(arr, currTag-bufSwaps, bufferStart, bufSwaps)
            self.lastBuffer = max(self.lastBuffer, bufSwaps)
    
    def smartCollectBackward(self, arr, start, end, target): # Interweaves build blocks and collect keys to achieve O(n) worst-case
        n = end-start
        runSize = self.minRun
        keysEnd = end
        keyCount = 1
        runEnd = end-1
        runStart = end - n%runSize
        if runStart == end:
            runStart -= runSize
        
        while runEnd > start:
            currRun = (runStart-start+runSize-1)//runSize
            
            keyPtr = keysEnd-1
            runPtr = runEnd-1
            
            while runPtr >= runStart:
                val = arr[runPtr]
                while keyPtr >= keysEnd-keyCount and arr[keyPtr] > val:
                    keyPtr -= 1
                if keyPtr < keysEnd-keyCount or arr[keyPtr] != val:
                    self.rotate(arr, runPtr+1, keysEnd-keyCount, keysEnd)
                    ofs = keysEnd-keyCount-(runPtr+1)
                    keysEnd -= ofs
                    keyPtr -= ofs
                    self.insertRight(arr, runPtr, keyPtr)
                    keyCount += 1
                    runPtr -= 1
                else:
                    while runEnd > start:
                        while runPtr >= runStart:
                            val = arr[runPtr]
                            while keyPtr >= keysEnd-keyCount and arr[keyPtr] > val:
                                keyPtr -= 1
                            
                            if keyPtr < keysEnd-keyCount or arr[keyPtr] != val:
                                break
                            
                            while runPtr >= runStart and arr[runPtr] == val:
                                runPtr -= 1
                        
                        if runPtr >= runStart or keyCount >= runSize and currRun%2 == 0:
                            break
                        
                        keyPtr = keysEnd-1
                        
                        runEnd = runStart
                        runStart = max(start, runEnd-runSize)
                        
                        currRun = (runStart-start+runSize-1)//runSize
                
                if keyCount >= target:
                    break
            
            if keyCount >= target:
                break
            
            if keyCount >= runSize and currRun%2 == 0:
                self.bufferStart = keysEnd-keyCount
                self.bufferLen = keyCount
                for left in range(start, runStart, 2*runSize):
                    mid = min(left+runSize, runStart)
                    right = min(left+2*runSize, runStart)
                    
                    if mid == right:
                        break
                    
                    skip, l, r = self.checkBounds(arr, left, mid, right, False)
                    
                    if skip:
                        continue
                    
                    self.mergeDecide(arr, l, mid, r, False)
                
                self.shellsort(arr, self.bufferStart, self.bufferStart+self.lastBuffer)
                self.lastBuffer = 0
                runSize *= 2
                
            
            runEnd = runStart
            runStart = max(start, runEnd-runSize)
        
        lastRotated = keysEnd-keyCount
        lastRotated -= (lastRotated-start)%runSize
        
        if keyCount >= target:
            self.rotate(arr, keysEnd-keyCount, keysEnd, end)
        else:
            dist = end-keysEnd
            self.blockSwapForward(arr, keysEnd-keyCount, keysEnd, dist)
            keyCount = self.floorPow2(keyCount)
        
        if keyCount < target and keyCount <= 4:
            keyCount = 0
        
        keysEnd -= keyCount
        regionEnd = end-keyCount
        self.bufferStart = regionEnd
        self.bufferLen = keyCount
        
        self.buildRuns(arr, lastRotated, regionEnd)
        mergeN = self.minRun
        while mergeN < runSize:
            for left in range(lastRotated, regionEnd, 2*mergeN):
                mid = min(left+mergeN, regionEnd)
                right = min(left+2*mergeN, regionEnd)
                
                if mid == right:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right, False)
                
                if skip:
                    continue
                
                self.mergeDecide(arr, l, mid, r, False)
            mergeN *= 2
        
        mergeN = runSize
        while mergeN <= keyCount:
            for left in range(start, regionEnd, 2*mergeN):
                mid = min(left+mergeN, regionEnd)
                right = min(left+2*mergeN, regionEnd)
                
                if mid == right:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right, False)
                
                if skip:
                    continue
                
                self.mergeDecide(arr, l, mid, r, False)
            mergeN *= 2
        
        self.minRun = max(runSize, mergeN//2)
        return keyCount
    
    def mergeDecide(self, arr, start, mid, end, right): # Helper
        lenA = mid-start
        lenB = end-mid
        
        if min(lenA, lenB) <= self.sizeSmall or min(lenA, lenB) > self.bufferLen:
            self.lazyStableMerge(arr, start, mid, end, right)
        else:
            self.mergeBuf(arr, start, mid, end, right)
    
    def mergeBuf(self, arr, start, mid, end, right): # Ensures merge direction matches buffer's capacity
        leftLen = mid-start
        rightLen = end-mid
        if leftLen <= rightLen:
            self.lastBuffer = max(self.lastBuffer, leftLen)
            self.mergeBufForward(arr, start, mid, end, right)
        else:
            self.lastBuffer = max(self.lastBuffer, rightLen)
            self.mergeBufBackward(arr, start, mid, end, right)
    
    def mergeExt(self, arr, start, mid, end, right):
        if mid-start <= end-mid:
            self.mergeExtForward(arr, start, mid, end, right)
        else:
            self.mergeExtBackward(arr, start, mid, end, right)
    
    def checkBounds(self, arr, start, mid, end, right): # Merge optimiser - skips where merges aren't needed and shrinks bounds
        if arr[mid-1] <= arr[mid]:
            return True, start, end
        
        start = self.expSearchForward(arr, arr[mid], start, mid, not right)
        end = self.expSearchBackward(arr, arr[mid-1], mid, end, right)
        
        if (arr[start] >= arr[end-1] if right else arr[start] > arr[end-1]):
            self.rotate(arr, start, mid, end)
            return True, start, end
        
        return False, start, end

    def mergeBufForward(self, arr, start, mid, end, right):
        bufferStart = self.bufferStart
        leftLen = mid-start
        self.blockSwapForward(arr, bufferStart, start, leftLen)
        l, r, dest = bufferStart, mid, start
        countLeft = countRight = 0
        minGallop = self.minGallop
        
        if right:
            while l < bufferStart+leftLen and r < end:
                while l < bufferStart+leftLen and r < end and (countLeft | countRight) < minGallop:
                    if arr[l] < arr[r]:
                        self.swap(arr, l, dest)
                        l += 1
                        countLeft += 1
                        countRight = 0
                    else:
                        self.swap(arr, r, dest)
                        r += 1
                        countRight += 1
                        countLeft = 0
                    dest += 1
                
                while l < bufferStart+leftLen and r < end:
                    countLeft = self.expSearchForward(arr, arr[r], l, bufferStart+leftLen, False)-l
                    if countLeft:
                        self.blockSwapForward(arr, l, dest, countLeft)
                        l += countLeft; dest += countLeft
                        if l >= bufferStart+leftLen:
                            break
                    self.swap(arr, r, dest)
                    r += 1; dest += 1
                    if r >= end:
                        break
                    countRight = self.expSearchForward(arr, arr[l], r, end, True)-r
                    if countRight:
                        self.blockSwapForward(arr, r, dest, countRight)
                        r += countRight; dest += countRight
                        if r >= end:
                            break
                    self.swap(arr, l, dest)
                    l += 1; dest += 1
                    if l >= bufferStart+leftLen:
                        break
                    
                    if (countLeft | countRight) < minGallop:
                        minGallop += 2
                        break
        else:
            while l < bufferStart+leftLen and r < end:
                while l < bufferStart+leftLen and r < end and (countLeft | countRight) < minGallop:
                    if arr[l] <= arr[r]:
                        self.swap(arr, l, dest)
                        l += 1
                        countLeft += 1
                        countRight = 0
                    else:
                        self.swap(arr, r, dest)
                        r += 1
                        countRight += 1
                        countLeft = 0
                    dest += 1
                
                while l < bufferStart+leftLen and r < end:
                    countLeft = self.expSearchForward(arr, arr[r], l, bufferStart+leftLen, True)-l
                    if countLeft:
                        self.blockSwapForward(arr, l, dest, countLeft)
                        l += countLeft; dest += countLeft
                        if l >= bufferStart+leftLen:
                            break
                    self.swap(arr, r, dest)
                    r += 1; dest += 1
                    if r >= end:
                        break
                    countRight = self.expSearchForward(arr, arr[l], r, end, False)-r
                    if countRight:
                        self.blockSwapForward(arr, r, dest, countRight)
                        r += countRight; dest += countRight
                        if r >= end:
                            break
                    self.swap(arr, l, dest)
                    l += 1; dest += 1
                    if l >= bufferStart+leftLen:
                        break
                    
                    if (countLeft | countRight) < minGallop:
                        minGallop += 2
                        break
        
        while l < bufferStart+leftLen:
            self.swap(arr, l, dest)
            l += 1
            dest += 1
        
        self.minGallop = minGallop
    
    def mergeBufBackward(self, arr, start, mid, end, right):
        bufferStart = self.bufferStart
        rightLen = end-mid
        self.blockSwapBackward(arr, bufferStart+rightLen, end, rightLen)
        l, r, dest = mid-1, bufferStart+rightLen-1, end-1
        countLeft = countRight = 0
        minGallop = self.minGallop
        
        if right:
            while l >= start and r >= bufferStart:
                while l >= start and r >= bufferStart and (countLeft | countRight) < minGallop:
                    if arr[r] > arr[l]:
                        self.swap(arr, r, dest)
                        r -= 1
                        countRight += 1
                        countLeft = 0
                    else:
                        self.swap(arr, l, dest)
                        l -= 1
                        countLeft += 1
                        countRight = 0
                    dest -= 1
                
                while l >= start and r >= bufferStart:
                    countLeft = l+1-self.expSearchBackward(arr, arr[r], start, l+1, False)
                    if countLeft:
                        self.blockSwapBackward(arr, l+1, dest+1, countLeft)
                        l -= countLeft; dest -= countLeft
                        if l < start:
                            break
                    self.swap(arr, r, dest)
                    r -= 1; dest -= 1
                    if r < bufferStart:
                        break
                    countRight = r+1-self.expSearchBackward(arr, arr[l], bufferStart, r+1, True)
                    if countRight:
                        self.blockSwapBackward(arr, r+1, dest+1, countRight)
                        r -= countRight; dest -= countRight
                        if r < bufferStart:
                            break
                    self.swap(arr, l, dest)
                    l -= 1; dest -= 1
                    if l < start:
                        break
                    
                    if (countLeft | countRight) < minGallop:
                        minGallop += 2
                        break
        else:
            while l >= start and r >= bufferStart:
                while l >= start and r >= bufferStart and (countLeft | countRight) < minGallop:
                    if arr[r] >= arr[l]:
                        self.swap(arr, r, dest)
                        r -= 1
                        countRight += 1
                        countLeft = 0
                    else:
                        self.swap(arr, l, dest)
                        l -= 1
                        countLeft += 1
                        countRight = 0
                    dest -= 1
                
                while l >= start and r >= bufferStart:
                    countLeft = l+1-self.expSearchBackward(arr, arr[r], start, l+1, True)
                    if countLeft:
                        self.blockSwapBackward(arr, l+1, dest+1, countLeft)
                        l -= countLeft; dest -= countLeft
                        if l < start:
                            break
                    self.swap(arr, r, dest)
                    r -= 1; dest -= 1
                    if r < bufferStart:
                        break
                    countRight = r+1-self.expSearchBackward(arr, arr[l], bufferStart, r+1, False)
                    if countRight:
                        self.blockSwapBackward(arr, r+1, dest+1, countRight)
                        r -= countRight; dest -= countRight
                        if r < bufferStart:
                            break
                    self.swap(arr, l, dest)
                    l -= 1; dest -= 1
                    if l < start:
                        break
                    
                    if countLeft < minGallop or countRight < minGallop:
                        minGallop += 2
                        break
        
        while r >= bufferStart:
            self.swap(arr, r, dest)
            r -= 1
            dest -= 1
        
        self.minGallop = minGallop
    
    def mergeExtForward(self, arr, start, mid, end, right):
        leftLen = mid-start
        buf = self.buffer
        self.arrayCopyForward(arr, start, buf, 0, leftLen)
        l, r, dest = 0, mid, start
        
        countLeft = countRight = 0
        minGallop = self.minGallop
        
        if right:
            while l < leftLen and r < end:
                while l < leftLen and r < end and (countLeft|countRight) < minGallop:
                    if buf[l] < arr[r]:
                        arr[dest] = buf[l]
                        l += 1
                        countLeft += 1
                        countRight = 0
                    else:
                        arr[dest] = arr[r]
                        r += 1
                        countRight += 1
                        countLeft = 0
                    dest += 1
                
                while l < leftLen and r < end:
                    countLeft = self.expSearchForward(buf, arr[r], l, leftLen, False)-l
                    if countLeft:
                        self.arrayCopyForward(buf, l, arr, dest, countLeft)
                        l += countLeft; dest += countLeft
                        if l >= leftLen:
                            break
                    arr[dest] = arr[r]
                    r += 1; dest += 1
                    if r >= end:
                        break
                    countRight = self.expSearchForward(arr, buf[l], r, end, True)-r
                    if countRight:
                        self.arrayCopyForward(arr, r, arr, dest, countRight)
                        r += countRight; dest += countRight
                        if r >= end:
                            break
                    arr[dest] = buf[l]
                    l += 1; dest += 1
                    if l >= leftLen:
                        break
                    
                    if (countLeft|countRight) < minGallop:
                        minGallop += 2
                        break
        else:
            while l < leftLen and r < end:
                while l < leftLen and r < end and (countLeft|countRight) < minGallop:
                    if buf[l] <= arr[r]:
                        arr[dest] = buf[l]
                        l += 1
                        countLeft += 1
                        countRight = 0
                    else:
                        arr[dest] = arr[r]
                        r += 1
                        countRight += 1
                        countLeft = 0
                    dest += 1
                
                while l < leftLen and r < end:
                    countLeft = self.expSearchForward(buf, arr[r], l, leftLen, True)-l
                    if countLeft:
                        self.arrayCopyForward(buf, l, arr, dest, countLeft)
                        l += countLeft; dest += countLeft
                        if l >= leftLen:
                            break
                    arr[dest] = arr[r]
                    r += 1; dest += 1
                    if r >= end:
                        break
                    countRight = self.expSearchForward(arr, buf[l], r, end, False)-r
                    if countRight:
                        self.arrayCopyForward(arr, r, arr, dest, countRight)
                        r += countRight; dest += countRight
                        if r >= end:
                            break
                    arr[dest] = buf[l]
                    l += 1; dest += 1
                    if l >= leftLen:
                        break
                    
                    if (countLeft|countRight) < minGallop:
                        minGallop += 2
                        break
        
        while l < leftLen:
            arr[dest] = buf[l]
            dest += 1
            l += 1
        
        self.minGallop = minGallop
    
    def mergeExtBackward(self, arr, start, mid, end, right):
        buf = self.buffer
        rightLen = end-mid
        self.arrayCopyBackward(arr, end, buf, rightLen, rightLen)
        l, r, dest = mid-1, rightLen-1, end-1
        countLeft = countRight = 0
        minGallop = self.minGallop
        
        if right:
            while l >= start and r >= 0:
                while l >= start and r >= 0 and (countLeft | countRight) < minGallop:
                    if buf[r] > arr[l]:
                        arr[dest] = buf[r]
                        r -= 1
                        countRight += 1
                        countLeft = 0
                    else:
                        arr[dest] = arr[l]
                        l -= 1
                        countLeft += 1
                        countRight = 0
                    dest -= 1
                
                while l >= start and r >= 0:
                    countLeft = l+1-self.expSearchBackward(arr, buf[r], start, l+1, False)
                    if countLeft:
                        self.arrayCopyBackward(arr, l+1, arr, dest+1, countLeft)
                        l -= countLeft; dest -= countLeft
                        if l < start:
                            break
                    arr[dest] = buf[r]
                    r -= 1; dest -= 1
                    if r < 0:
                        break
                    countRight = r+1-self.expSearchBackward(buf, arr[l], 0, r+1, True)
                    if countRight:
                        self.arrayCopyBackward(buf, r+1, arr, dest+1, countRight)
                        r -= countRight; dest -= countRight
                        if r < 0:
                            break
                    arr[dest] = arr[l]
                    l -= 1; dest -= 1
                    if l < start:
                        break
                    
                    if (countLeft | countRight) < minGallop:
                        minGallop += 2
                        break
        else:
            while l >= start and r >= 0:
                while l >= start and r >= 0 and (countLeft | countRight) < minGallop:
                    if buf[r] >= arr[l]:
                        arr[dest] = buf[r]
                        r -= 1
                        countRight += 1
                        countLeft = 0
                    else:
                        arr[dest] = arr[l]
                        l -= 1
                        countLeft += 1
                        countRight = 0
                    dest -= 1
                
                while l >= start and r >= 0:
                    countLeft = l+1-self.expSearchBackward(arr, buf[r], start, l+1, True)
                    if countLeft:
                        self.arrayCopyBackward(arr, l+1, arr, dest+1, countLeft)
                        l -= countLeft; dest -= countLeft
                        if l < start:
                            break
                    arr[dest] = buf[r]
                    r -= 1; dest -= 1
                    if r < 0:
                        break
                    countRight = r+1-self.expSearchBackward(buf, arr[l], 0, r+1, False)
                    if countRight:
                        self.arrayCopyBackward(buf, r+1, arr, dest+1, countRight)
                        r -= countRight; dest -= countRight
                        if r < 0:
                            break
                    arr[dest] = arr[l]
                    l -= 1; dest -= 1
                    if l < start:
                        break
                    
                    if (countLeft | countRight) < minGallop:
                        minGallop += 2
                        break
        
        while r >= 0:
            arr[dest] = buf[r]
            r -= 1
            dest -= 1
        
        self.minGallop = minGallop
    
    def lazyStableMerge(self, arr, start, mid, end, right):
        bufferLen = 0 if self.buffer is None or min(mid-start, end-mid) < self.sizeSmall else len(self.buffer)
        
        if mid-start <= end-mid:
            s = start
            l = mid

            while l+1-s > bufferLen and l < end:
                if right:
                    cmp = arr[s] >= arr[l]
                else:
                    cmp = arr[s] > arr[l]
                
                if cmp:
                    p = self.expSearchForward(arr, arr[s], l, end, right)
                    self.rotate(arr, s, l, p)
                    s += p-l
                    l = p
                else:
                    s += 1
            if l+1-s > 0 and l < end:
                self.mergeExt(arr, s, l, end, right)
        else:
            s = end-1
            l = mid-1

            while s-l > bufferLen and l >= start:
                if right:
                    cmp = arr[l] >= arr[s]
                else:
                    cmp = arr[l] > arr[s]
                
                if cmp:
                    p = self.expSearchBackward(arr, arr[s], start, l, not right)
                    self.rotate(arr, p, l+1, s+1)
                    s -= l+1-p
                    l = p-1
                else:
                    s -= 1
            
            if s-l > 0 and l >= start:
                self.mergeExt(arr, start, l+1, s+1, right)
    
    def blockMerge(self, arr, start, mid, end):
        tagsStart = self.tagsStart
        bufferStart = self.bufferStart
        blockLen = self.blockLen
        
        leftLen, rightLen = mid-start, end-mid
        
        leftBlocks = leftLen//blockLen
        rightBlocks = rightLen//blockLen
        
        blockCount = leftBlocks+rightBlocks
        frag = (end-start)-blockCount*blockLen
        
        if not self.smartTagSort:
            self.shellsort(arr, tagsStart, tagsStart+blockCount)
        
        midTag = arr[tagsStart+leftBlocks]
        
        self.smartBlockSelect(arr, start, blockCount, blockLen, leftBlocks, arr)
        self.mergeBlocks(arr, start, blockCount, blockLen, frag, midTag, arr)
        
        if self.smartTagSort:
            self.tagSort(arr, tagsStart, blockCount, midTag)
    
    def smartBlockSelect(self, arr, start, blockCount, blockLen, blockB, tags):
        tagsStart = self.tagsStart
        endB = blockB+1
        
        for i in range(blockCount):
            minIdx = i
            minVal = arr[start+(i+1)*blockLen-1]
            
            for j in range(max(i+1, blockB), endB):
                blockVal = arr[start+(j+1)*blockLen-1]
                
                if blockVal < minVal or (blockVal == minVal and tags[tagsStart+j] < tags[tagsStart+minIdx]):
                    minVal = blockVal
                    minIdx = j
            
            if minIdx != i:
                self.blockSwapForward(arr, start+i*blockLen, start+minIdx*blockLen, blockLen)
                self.swap(tags, tagsStart+i, tagsStart+minIdx)
                
                if endB < blockCount and minIdx == endB-1:
                    endB += 1
    
    def mergeBlocks(self, arr, start, blockCount, blockLen, remainder, midTag, tags):
        tagsStart = self.tagsStart
        fragStart = start
        right = tags[tagsStart] >= midTag
        for i in range(1, blockCount):
            if right ^ (tags[tagsStart+i] >= midTag):
                next = start+i*blockLen
                fragStart = self.expSearchForward(arr, arr[next], fragStart, next, not right) # optional
                nextEnd = self.expSearchBackward(arr, arr[next-1], next, next+blockLen, right) # mandatory
                self.mergeDecide(arr, fragStart, next, nextEnd, right)
                fragStart = nextEnd
                right = not right
        
        if not right and remainder != 0:
            lastFrag = start+blockCount*blockLen
            self.mergeDecide(arr, fragStart, lastFrag, lastFrag+remainder, right)
    
    def rotate(self, arr, a, m, e): # Trinity rotation
        buf = self.buffer
        
        lenA = m-a
        lenB = e-m

        if lenA < 1 or lenB < 1:
            return
        
        if lenA == 1:
            self.insertRight(arr, a, e-1)
            return
        
        if buf is not None and lenA <= len(buf):
            self.arrayCopyForward(arr, a, buf, 0, lenA)
            self.arrayCopyForward(arr, m, arr, a, lenB)
            self.arrayCopyForward(buf, 0, arr, a+lenB, lenA)
            return
        
        if lenB == 1:
            self.insertLeft(arr, e-1, a)
            return
        
        if buf is not None and lenB <= len(buf):
            self.arrayCopyBackward(arr, e, buf, lenB, lenB)
            self.arrayCopyBackward(arr, m, arr, e, lenA)
            self.arrayCopyBackward(buf, lenB, arr, a+lenB, lenB)
            return
        
        if max(lenA, lenB) % min(lenA, lenB) == 0: # New - Scroll instead of plain block swap
            if lenA <= lenB:
                self.blockSwapForward(arr, a, m, lenB)
            else:
                self.blockSwapBackward(arr, m, e, lenA)
            return

        b = m-1
        c = m
        d = e-1

        tmp = 0
        spd = UniV_getSpeed() # Sequential writes look slower in visualisers even if they're faster in the CPU. Speed up to compensate.
        UniV_setSpeed(spd*4)
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

    def redistBuffer(self, arr, start, mid, end): # Modified block swap merge - original implementation from https://sortingalgos.miraheze.org/wiki/Rotate_Merge_Sort#Block-Swap_Merge
        leftLen = mid-start
        rightLen = end-mid
        while min(leftLen, rightLen) > max(self.sizeSmall, 0 if self.buffer is None else len(self.buffer)):
            leftLen = mid-start
            rightLen = end-mid
            
            forward = leftLen <= rightLen
            
            if forward:
                if arr[start] > arr[mid+leftLen]:
                    pos = self.expSearchForward(arr, arr[start], mid+leftLen, end, False)
                    self.rotate(arr, start, mid, pos)
                    dist = pos-mid
                    start += dist+1
                    mid += dist
                    leftLen -= 1
                    continue
            else:
                if arr[end-1] < arr[mid-rightLen]:
                    pos = self.expSearchBackward(arr, arr[end-1], start, mid-rightLen, True)
                    self.rotate(arr, pos, mid, end)
                    dist = mid-pos
                    end -= dist+1
                    mid -= dist
                    rightLen -= 1
                    continue
            
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
            
            if forward:
                self.lazyStableMerge(arr, start, mid-d, mid, False)
                start = mid
                mid += d
            else:
                self.lazyStableMerge(arr, mid, mid+d, end, False)
                end = mid
                mid -= d
        
        self.lazyStableMerge(arr, start, mid, end, False)
    
    def fulcrumPartition(self, arr, head, tail):
        mid = head+(tail-head+1)//2
        if arr[head] > arr[mid]:
            self.swap(arr, head, mid)
        if arr[mid] > arr[tail]:
            self.swap(arr, mid, tail)
        if arr[head] > arr[mid]:
            self.swap(arr, head, mid)
        self.swap(arr, mid, head)
        
        pivot = arr[head]
        
        while True:
            while arr[tail] > pivot: 
                tail -= 1
            if head >= tail:
                arr[head] = pivot
                return head
            arr[head] = arr[tail]
            head += 1

            while arr[head] <= pivot:
                head += 1
            if head >= tail:
                arr[tail] = pivot
                return tail
            arr[tail] = arr[head]
            tail -= 1

    def quickselect(self, arr, k, start, end):
        head = start
        tail = end-1

        while True:
            pos = self.fulcrumPartition(arr, head, tail)
            if pos == k:
                return arr[pos]
            elif pos > k:
                tail = pos-1
            else:
                head = pos+1
    
    def trySmallSort(self, arr, start, end): # Run-aware exponential insertion sort (consider lazy stable insertion?)
        if end-start <= 2*self.sizeSmall:
            hint = self.countRun(arr, start, end)
            self.expSort(arr, start, end, hint)
            return True
        return False
    
    def blockSetup(self, arr, start, end):
        n = end-start
        self.blockLen = self.sqrtPow2(n)
        
        while 2*self.blockLen <= self.mem:
            self.blockLen *= 2
        
        if self.blockLen > self.mem:
            self.buffer = None if self.mem == 0 else Array(min(self.mem, self.rotateMaxAux))
        else:
            self.buffer = Array(self.blockLen)
        
        if self.blockLen == self.sizeSmall or self.buffer is not None and len(self.buffer) == self.blockLen:
            bufferTarget = 0
        else:
            bufferTarget = self.blockLen
        
        self.tagsLen = n//self.blockLen
        target = bufferTarget+self.tagsLen
        
        self.keys = self.smartCollectBackward(arr, start, end, target)
        
        if self.keys == 0:
            return
        
        self.tagsStart = end-self.keys
        if self.keys < target:
            self.blockLen = self.sqrtPow2(2*self.minRun)
            self.tagsLen = (2*self.minRun)//self.blockLen
        
        self.bufferStart = self.tagsStart+self.tagsLen
        self.bufferLen = max(0, self.keys-self.tagsLen)
        
        self.recheck = (self.keys < target)
        self.smartTagSort = (self.blockLen > self.sizeSmall and not self.recheck)
        
        if self.smartTagSort:
            if self.tagsStart+self.lastBuffer > self.bufferStart:
                self.quickselect(arr, self.bufferStart, self.tagsStart, self.tagsStart+self.lastBuffer)
                self.shellsort(arr, self.tagsStart, self.bufferStart)
                self.lastBuffer -= self.tagsLen
            else:
                self.shellsort(arr, self.tagsStart, self.tagsStart+self.lastBuffer)
                self.lastBuffer = 0
    
    def recalcBlockLen(self, mergeN):
        root = self.sqrtPow2(2*mergeN)
        
        if 2*root+(2*mergeN)//(2*root) < self.keys:
            root *= 2
        
        tagCount = (2*mergeN)//root
        if tagCount < self.keys:
            self.bufferLen = self.keys-tagCount
            self.bufferStart = self.tagsStart+tagCount
        else:
            while (2*mergeN)//root > self.keys:
                root *= 2
            self.bufferLen = 0
        self.blockLen = root
    
    def lazyStableLoop(self, arr, start, end):
        n = end-start
        mergeN = self.minRun
        while mergeN < n:
            for left in range(start, end, 2*mergeN):
                mid = min(left+mergeN, end)
                right = min(mid+mergeN, end)
                
                if mid == right:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right, False)
                if skip:
                    continue
                
                self.lazyStableMerge(arr, l, mid, r, False)
            mergeN *= 2
    
    def blockMergeLoop(self, arr, start, end):
        regionEnd = end-self.keys
        n = regionEnd-start
        
        mergeN = self.minRun
        while mergeN < n:
            if self.recheck:
                self.recalcBlockLen(mergeN)
        
            for left in range(start, regionEnd, 2*mergeN):
                mid = min(left+mergeN, regionEnd)
                right = min(mid+mergeN, regionEnd)
                
                if mid == right:
                    break
                
                skip, l, r = self.checkBounds(arr, left, mid, right, False)
                if skip:
                    continue
                
                if mid-l > self.blockLen:
                    l -= (l-start)%self.blockLen
                
                if min(mid-l, r-mid) <= self.blockLen:
                    self.mergeDecide(arr, l, mid, r, False)
                else:
                    self.blockMerge(arr, l, mid, r)
            mergeN *= 2
        
        if self.smartTagSort:
            self.shellsort(arr, self.bufferStart, self.bufferStart+self.lastBuffer)
        else:
            self.shellsort(arr, regionEnd, end)
        
        self.redistBuffer(arr, start, regionEnd, end)
        
    def sort(self, arr, start, end):
        n = end-start
        
        if self.trySmallSort(arr, start, end):
            return
        
        if self.buildRuns(arr, start, end) < 2: # Only one run was found. No need to do anything else, so exit early.
            return
        
        if n <= self.sizeSmall*4 or self.mem >= n//2:
            self.buffer = Array(n//2)
            self.lazyStableLoop(arr, start, end)
            return
        
        self.blockSetup(arr, start, end)
        
        if self.keys == 0:
            self.lazyStableLoop(arr, start, end)
            return
        
        regionEnd = self.tagsStart
        
        self.blockMergeLoop(arr, start, end)
    
@Sort("Block Merge Sorts", "Miausort (Old)", "Miausort (Old)")
def miauSortRun(arr):
    n = len(arr)
    mem = UniV_getUserInput("Enter memory (0 to -2 for different strategies)", "0", parseInt)
    if mem == -1:
        mem = pow2Sqrt(n)
    elif mem == -2:
        mem = n//2
    
    MiauSorter = MiauSortOld(mem)
    MiauSorter.sort(arr, 0, len(arr))
