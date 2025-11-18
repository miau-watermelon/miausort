class Miausort:
    minRun = 16
    minGallop = 7
    
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
                if currEnd < end and arr[currEnd-1] >= arr[currEnd]:
                    runs += 1
                break
            
            if runLen < minRun:
                ext = max(currEnd-minRun, start)
                self.expSort(arr, ext, currEnd)
                if currEnd < end and arr[currEnd-1] >= arr[currEnd]:
                    runs += 1
                currEnd = ext
                continue
            
            currEnd = currStart+runLen%minRun
            if currEnd < end and arr[currEnd-1] >= arr[currEnd]:
                runs += 1
            
            if runLen%minRun == 0:
                continue
            
            ext = max(currEnd-minRun, start)
            self.expSort(arr, ext, currEnd)
            if currEnd < end and arr[currEnd-1] >= arr[currEnd]:
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
    
    def expSort(self, arr, start, end):
        speed = UniV_getSpeed()
        UniV_setSpeed(speed*3)
        for i in range(start+1, end):
            self.insert(arr, i, self.expSearchBackward(arr, arr[i], start, i, True))
        UniV_setSpeed(speed)
    
    def shellsort(self, arr, start, end): # Only used for sorting buffers as they contain all unique values
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
        
    def smartCollect(self, arr, start, end, target):
        n = end-start
        runSize = self.minRun
        keysStart = start
        keyCount = 0
        runStart = start+(n%runSize or runSize)-runSize
        lastRotated = start
        lastFixed = start
        while runStart < end and keyCount < target:
            runEnd = min(runStart+runSize, end)
            currRun = (end-runEnd+runSize-1)//runSize
            j = max(runStart, start)
            searchStart = keysStart
            while j < runEnd:
                val = arr[j]
                pos = self.expSearchForward(arr, val, searchStart, keysStart + keyCount, False)
                searchStart = pos
                
                if pos == keysStart+keyCount or arr[pos] != val:
                    self.insert(arr, j, pos)
                    keyCount += 1
                    j += 1
                
                else:
                    breakOuter = False
                    while runStart < end and not breakOuter:
                        while j < runEnd:
                            val = arr[j]
                            pos = self.expSearchForward(arr, val, searchStart, keysStart+keyCount, False)
                            searchStart = pos
                            if pos == keysStart+keyCount or arr[pos] != val:
                                breakOuter = True
                                break
                            j = self.expSearchForward(arr, arr[pos], j, runEnd, True)
                        
                        if breakOuter:
                            self.rotate(arr, keysStart, keysStart+keyCount, min(j,end))
                            ofs = j-(keysStart+keyCount)
                            searchStart += ofs
                            keysStart += ofs
                            lastRotated = min(runEnd+runSize, end)
                            break
                        
                        if keyCount >= runSize and currRun % 2 == 0:
                            self.rotate(arr, keysStart, keysStart+keyCount, min(j,end))
                            ofs = j-(keysStart+keyCount)
                            searchStart += ofs
                            keysStart += ofs
                            lastRotated = min(runEnd+runSize, end)
                            break
                        
                        if j < end:
                            searchStart = self.expSearchForward(arr, arr[j], keysStart, keysStart+keyCount, False)
                        
                        runStart = runEnd
                        runEnd = min(runStart+runSize, end)
                        while runEnd < end and arr[runEnd-1] <= arr[runEnd]:
                            runEnd = min(runEnd+runSize, end)
                        currRun = (end-runEnd+runSize-1)//runSize
                
                if keyCount >= target:
                    break
            
            if keyCount >= target:
                break
            
            if keyCount >= runSize and currRun % 2 == 0:
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
                
                fixStart = keysStart-runSize+(end-keysStart)%runSize
                if (end-fixStart+runSize-1)//runSize % 2 != 0:
                    fixStart -= runSize
                
                if fixStart > lastFixed:
                    self.buildRunsBackward(arr, lastFixed, fixStart)
                    mergeN = self.minRun
                    while mergeN < runSize:
                        for right in range(fixStart, lastFixed, -2*mergeN):
                            mid = max(lastFixed, right-mergeN)
                            left = max(lastFixed, right-2*mergeN)
                            
                            if left == mid:
                                break
                            
                            skip, l, r = self.checkBounds(arr, left, mid, right)
                            if skip:
                                continue
                            self.mergeBuf(arr, l, mid, r)
                        mergeN *= 2
                    lastFixed = fixStart
                
                for right in range(lastFixed, start, -2*runSize):
                    mid = max(start, right-runSize)
                    left = max(start, right-2*runSize)
                    
                    if left == mid:
                        break
                    
                    skip, l, r = self.checkBounds(arr, left, mid, right)
                    if skip:
                        continue
                    self.mergeBuf(arr, l, mid, r)
                
                self.shellsort(arr, keysStart, keysStart+runSize)
                runSize *= 2
            
            runStart = runEnd
        
        self.rotate(arr, start, keysStart, keysStart+keyCount)
        lastFixed += keyCount
        keysEnd = start+keyCount
        lastRotated += (end-lastRotated)%runSize
        self.buildRunsBackward(arr, lastFixed, lastRotated)
        
        self.bufferStart = start
        mergeN = self.minRun
        while mergeN < runSize:
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
        while mergeN < keyCount:
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
        
        self.minRun = mergeN//2
        return keyCount
    
    def mergeDecide(self, arr, start, mid, end):
        lenA = mid-start
        lenB = end-mid
        
        if min(lenA, lenB) <= self.buffer:
            self.mergeBuf(arr, start, mid, end)
        else:
            self.lazyStableMerge(arr, start, mid, end, False)
    
    def mergeBuf(self, arr, start, mid, end): # Ensures merge direction matches buffer's capacity
        if mid-start < end-mid:
            self.mergeBufForward(arr, start, mid, end)
        else:
            self.mergeBufBackward(arr, start, mid, end)
    
    def checkBounds(self, arr, start, mid, end):
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
        self.blockSwapForward(arr, start, bufferStart, leftLen) # Move half into buffer

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
    
    def lazyStableMerge(self, arr, start, mid, end, right):
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
        
        self.shellsort(arr, tagsStart, tagsStart+blockCount)
        
        midTag = arr[tagsStart+blockCountA]
        
        self.blockSelectSort(arr, blockStart, blockCount, blockLen, blockCountA, midTag)
        self.cleanupBlocks(arr, start, end, blockCount, blockLen, remainderA, remainderB, midTag)
    
    def blockSelectSort(self, arr, start, blockCount, blockLen, blockB, midTag):
        tagsStart = self.tagsStart
        b_end = blockB+1 # Only look at blocks if the min is guaranteed to be here

        for i in range(blockCount):
            minIdx = i

            if arr[tagsStart+i] < midTag: # Determining value used to compare blocks based on which half they came from to preserve stability
                minVal = arr[start+i*blockLen] # First item of A block
            else:
                minVal = arr[start+(i+1)*blockLen-1] # Last item of B block

            for j in range(max(blockB,i+1), b_end):
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

                if b_end < blockCount: # Only increase range being checked when a swap has been made
                    b_end += 1
    
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
        while i >= 0: # Doing this backwards is the simplest way to do it stably (I was not about to spend four years learning fragment logic)
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

        if lenA == 1: # Mini bridge rotate — temporary and will be replaced with a full bridge rotation when I re-add auxiliary memory
            self.insert(arr, a, e-1)
            return
        
        if lenB == 1:
            self.insert(arr, e-1, a)
            return

        b = m-1
        c = m
        d = e-1

        tmp = 0
        
        speed = UniV_getSpeed()
        UniV_setSpeed(speed*2)
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
        UniV_setSpeed(speed)
        if a < d:
            self.reverse(arr, a, d+1)
    
    def sort(self, arr, start, end):
        n = end-start
        
        if n <= 2*self.minRun:
            self.expSort(arr, start, end)
            return
        
        if self.buildRunsBackward(arr, start, end) < 2:
            return
        
        self.blockLen = self.sqrtPow2(n)
        
        bufferTarget = self.blockLen
        tagTarget = n//bufferTarget
        
        target = bufferTarget+tagTarget
        keys = self.smartCollect(arr, start, end, target)
        n -= keys
        if keys >= target:
            self.tagsStart = start
            self.tags = n//self.blockLen
            self.bufferStart = self.tagsStart+self.tags
            self.buffer = self.blockLen
            recheck = False
        else:
            self.tagsStart = start
            self.blockLen = max(int((end-start)**0.5), self.blockSize((end-start), keys))
            self.tags = (2*self.minRun)//self.blockLen
            self.bufferStart = self.tagsStart+self.tags
            self.buffer = keys-self.tags
            recheck = True
        regionStart = start+keys
        mergeN = self.minRun
        while mergeN < end-regionStart:
            if recheck:
                self.blockLen = max(int((2*mergeN)**0.5), self.blockSize((2*mergeN), keys-8))
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
                
                if min(mid-l, r-mid) <= self.blockLen:
                    self.mergeDecide(arr, l, mid, r)
                else:
                    
                    self.blockMerge(arr, l, mid, r)
            mergeN *= 2
        
        self.shellsort(arr, start, start+keys)
        self.redistBuffer(arr, start, start+keys, end)
    
    def redistBuffer(self, arr, start, mid, end):
        rPos = self.expSearchForward(arr, arr[start], mid, end, False)
        self.rotate(arr, start, mid, rPos)
        
        dist = rPos-mid
        start += dist
        mid += dist
        
        start1 = start+(mid-start)//2
        rPos = self.expSearchForward(arr, arr[start1], mid, end, False)
        self.rotate(arr, start1, mid, rPos)
        
        dist = rPos-mid
        start1 += dist
        mid += dist
        
        self.lazyStableMerge(arr, start, start1-dist, start1, False)
        self.lazyStableMerge(arr, start1, mid, end, False)

@Sort("Block Merge Sorts", "Stackless Miausort", "Stackless Miausort")
def miauSort(arr):

    Miausort().sort(arr, 0, len(arr))
