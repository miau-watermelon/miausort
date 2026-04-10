"""
MIT License

Copyright (c) 2025-2026 miau

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

class MiauSort:
    smallMerge = 8
    minRun = 16
    lastUsed = 0
    
    def sqrtPow2(self, n):
        b = self.minRun
        while b * b < n - b:
            b *= 2
        return b
    
    def floorPow2(self, n):
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n - (n >> 1)
    
    def bSearch(self, arr, x, a, b, left):
        if left:
            while a < b:
                m = a + (b - a) // 2
                if arr[m] < x:
                    a = m + 1
                else:
                    b = m
        else:
            while a < b:
                m = a + (b - a) // 2
                if arr[m] <= x:
                    a = m + 1
                else:
                    b = m
        return a
    
    def expSearchFW(self, arr, x, a, b, left):
        i = 0
        i1 = 0
        
        if left:
            while a + i < b and arr[a + i] < x:
                i1 = i
                i = i * 2 + 1
        else:
            while a + i < b and arr[a + i] <= x:
                i1 = i
                i = i * 2 + 1
        
        return self.bSearch(arr, x, a + i1, min(a + i, b), left)
    
    def expSearchBW(self, arr, x, a, b, left):
        i = 0
        i1 = 0
        b1 = b-1
        
        if left:
            while b1 - i >= a and arr[b1 - i] >= x:
                i1 = i
                i = i * 2 + 1
        else:
            while b1 - i >= a and arr[b1 - i] > x:
                i1 = i
                i = i * 2 + 1
        
        return self.bSearch(arr, x, max(b1 - i, a), b - i1, left)
    
    def swap(self, arr, a, b):
        tmp = arr[a]
        arr[a] = arr[b]
        arr[b] = tmp
    
    def reverse(self, arr, a, b):
        b -= 1
        while a < b:
            self.swap(arr, a, b)
            a += 1
            b -= 1
    
    def blockSwap(self, arr, a, b, n):
        for i in range(n):
            self.swap(arr, a + i, b + i)
    
    def blockSwapBW(self, arr, a, b, n):
        a -= 1; b -= 1
        for i in range(n):
            self.swap(arr, a - i, b - i)
    
    def triBlockSwap(self, arr, a, b, c, n): # Swap ABC -> BCA
        for i in range(n):
            tmp = arr[a + i]
            arr[a + i] = arr[b + i]
            arr[b + i] = arr[c + i]
            arr[c + i] = tmp
    
    def insertLeft(self, arr, a, b):
        tmp = arr[a]
        while a > b:
            arr[a] = arr[a - 1]
            a -= 1
        arr[b] = tmp
    
    def insertRight(self, arr, a, b):
        tmp = arr[a]
        while a < b:
            arr[a] = arr[a + 1]
            a += 1
        arr[b] = tmp
    
    def rotate(self, arr, a, m, e): # Trinity rotation
        lenA = m - a
        lenB = e - m

        if lenA < 1 or lenB < 1:
            return
        
        if lenA == 1 or lenB == 1:
            if lenA <= lenB:
                self.insertRight(arr, a, e - 1)
            else:
                self.insertLeft(arr, e - 1, a)
            return
        
        if max(lenA, lenB) % min(lenA, lenB) == 0:
            if lenA <= lenB:
                self.blockSwap(arr, a, m, lenB)
            else:
                self.blockSwapBW(arr, m, e, lenA)
            return
        
        b = m - 1
        c = m
        d = e - 1
        
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
        
        if a < d:
            self.reverse(arr, a, d+1)
    
    def bSort(self, arr, a, b, h):
        for i in range(h, b):
            if arr[i - 1] > arr[i]:
                self.insertLeft(arr, i, self.bSearch(arr, arr[i], a, i, False))
    
    def shellsort(self, arr, start, end): # Only used for sorting buffers since they contain fully distinct values
        gaps = [1073790977, 268460033, 67121153, 16783361, 4197377, 1050113, 262913, 65921, 16577, 4193, 1073, 281, 77, 23, 8, 3, 1]

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
    
    def scrollMerge(self, arr, p, a, m, left):
        i = m
        if left:
            while a < m:
                if arr[a] <= arr[i]:
                    self.swap(arr, a, p)
                    a += 1
                else:
                    self.swap(arr, i, p)
                    i += 1
                p += 1
        else:
            while a < m:
                if arr[a] < arr[i]:
                    self.swap(arr, a, p)
                    a += 1
                else:
                    self.swap(arr, i, p)
                    i += 1
                p += 1
        return i
    
    def tailMerge(self, arr, p, a, m, b, bufPos, bLen):
        i = m
        
        while a < m and i < b:
            if arr[a] <= arr[i]:
                self.swap(arr, a, p)
                a += 1
            else:
                self.swap(arr, i, p)
                i += 1
            p += 1
        
        if a < m:
            if a > p:
                self.blockSwap(arr, p, a, m - a)
            self.blockSwap(arr, bufPos, b-bLen, bLen)
            
            return
        
        a = 0
        while a < bLen and i < b:
            if arr[bufPos + a] <= arr[i]:
                self.swap(arr, bufPos + a, p)
                a += 1
            else:
                self.swap(arr, i, p)
                i += 1
            p += 1
        
        self.blockSwap(arr, p, bufPos + a, bLen - a)
    
    def blockSelectSort(self, arr, a, bCount, bLen, lCount, t):
        midTag = lCount
        
        k = lCount+1
        j = 0
        while j < k - 1:
            minIdx = j
            minVal = arr[a + (j + 1) * bLen - 1]
            for i in range(max(j + 1, lCount-1), k):
                bVal = arr[a + (i + 1) * bLen - 1]
                if bVal < minVal or (bVal == minVal and arr[t + i] < arr[t + minIdx]):
                    minIdx = i
                    minVal = bVal
            
            if minIdx != j:
                self.blockSwap(arr, a + j * bLen, a + minIdx * bLen, bLen)
                self.swap(arr, t + j, t + minIdx)
                
                if k < bCount and minIdx == k - 1:
                    k += 1
            
            if minIdx == midTag:
                midTag = j
            
            j += 1
        
        return t + midTag
    
    def blockMerge(self, arr, a, m, b, bLen, buf, t):
        a1 = a + bLen
        lenL = m - a1
        lenR = b - m
        
        countL = lenL // bLen
        countR = lenR // bLen
        rem = lenR & (bLen - 1)
        bCount = countL+countR
        
        self.triBlockSwap(arr, buf, m - bLen, a, bLen)
        self.insertRight(arr, t, t + countL - 1)
        
        midTag = self.blockSelectSort(arr, a1, bCount, bLen, countL, t)
        
        f = a1
        left = arr[t] < arr[midTag]
        for i in range(1, bCount):
            if left ^ (arr[t + i] < arr[midTag]):
                nxt = a1 + i * bLen
                f = self.scrollMerge(arr, f - bLen, f, nxt, left)
                left = not left
        
        if left:
            self.tailMerge(arr, f - bLen, f, b - rem, b, buf, bLen)
        else:
            self.tailMerge(arr, f - bLen, f, f, b, buf, bLen)
        
        self.swap(arr, midTag, buf)
        bufSwaps = 1
        for c in range(midTag + 1, t + bCount):
            if arr[c] < arr[buf]:
                self.swap(arr, c, c - bufSwaps)
            else:
                self.swap(arr, c, buf + bufSwaps)
                bufSwaps += 1
        
        self.blockSwap(arr, t + bCount - bufSwaps, buf, bufSwaps)
        self.lastUsed = max(self.lastUsed, bLen)
    
    def lazyMerge(self, arr, a, m, b, left):
        if m - a <= b - m:
            s = a
            l = m

            while s < l and l < b:
                if left:
                    cmp = arr[s] > arr[l]
                else:
                    cmp = arr[s] >= arr[l]
                
                if cmp:
                    p = self.expSearchFW(arr, arr[s], l, b, left)
                    self.rotate(arr, s, l, p)
                    s += p - l
                    l = p
                else:
                    s += 1
        else:
            s = b - 1
            l = m - 1

            while s > l and l >= a:
                if left:
                    cmp = arr[l] > arr[s]
                else:
                    cmp = arr[l] >= arr[s]
                
                if cmp:
                    p = self.expSearchBW(arr, arr[s], a, l, not left)
                    self.rotate(arr, p, l + 1, s + 1)
                    s -= l + 1 - p
                    l = p - 1
                else:
                    s -= 1
    
    def blockMergeNoBuf(self, arr, a, m, b, bLen, t):
        lenL = m - a
        lenR = b - m
        
        countL = lenL // bLen
        countR = lenR // bLen
        rem = lenR & (bLen - 1)
        bCount = countL + countR
        
        midTag = self.blockSelectSort(arr, a, bCount, bLen, countL, t)
        
        f = a
        left = arr[t] < arr[midTag]
        for i in range(1, bCount):
            if left ^ (arr[t + i] < arr[midTag]):
                nxt = a + i * bLen
                nxtB = self.expSearchBW(arr, arr[nxt-1], nxt, nxt + bLen, left)
                self.lazyMerge(arr, f, nxt, nxtB, left)
                f = nxtB
                left = not left
        
        if left and rem > 0:
            self.lazyMerge(arr, f, b-rem, b, True)
        
        for c in range(midTag+1, t + bCount):
            if arr[c] < arr[midTag]:
                self.insertLeft(arr, c, midTag)
                midTag += 1
    
    def shrinkBounds(self, arr, a, m, b):
        if arr[m - 1] <= arr[m]:
            return -1, b
        
        a = self.expSearchFW(arr, arr[m], a, m, False)
        b = self.expSearchBW(arr, arr[m - 1], m, b, True)
        
        if arr[a] > arr[b - 1]:
            self.rotate(arr, a, m, b)
            return -1, b
        
        return a, b
    
    def mergeFW(self, arr, a, m, b, p):
        L = m - a
        self.lastUsed = max(self.lastUsed, L)
        self.blockSwap(arr, a, p, L)
        i = m
        while L > 0 and i < b:
            if arr[p] <= arr[i]:
                self.swap(arr, p, a)
                p += 1
                L -= 1
            else:
                self.swap(arr, i, a)
                i += 1
            a += 1
        
        self.blockSwap(arr, a, p, L)
    
    def mergeBW(self, arr, a, m, b, p):
        R = b - m
        self.lastUsed = max(self.lastUsed, R)
        self.blockSwap(arr, m, p, R)
        b -= 1
        i = m - 1
        while R > 0 and i >= a:
            if arr[p + R - 1] >= arr[i]:
                self.swap(arr, p + R - 1, b)
                R -= 1
            else:
                self.swap(arr, i, b)
                i -= 1
            b -= 1
        
        self.blockSwap(arr, p, b - R + 1, R)
    
    def countRun(self, arr, a, b):
        i = a + 1
        while i < b and arr[i - 1] <= arr[i]: # Scan for ascending/equal run
            i += 1
        
        if i >= b or i > a + 1 and arr[a] < arr[i - 1]: # If ascending, return
            return i
        
        p = arr[a]
        self.reverse(arr, a, i) # Flip current equal segment to invert stability
        s = i
        
        while s < b and arr[s] < p: # Only continue if the next segment is less than the previous
            c = arr[s]
            i += 1
            while i < b and arr[i] == c:
                i += 1
            
            self.reverse(arr, s, i) # Invert stability
            p = c
            s = i
        self.reverse(arr, a, i) # Reverse entire run
        return i
    
    def buildRuns(self, arr, a, b, r):
        mono = True
        while a < b:
            curr = self.countRun(arr, a, b)
            if curr == b:
                break
            
            mono = False
            
            rLen = curr - a
            
            if rLen < r:
                ext = min(a + r, b)
                self.bSort(arr, a, ext, curr)
                a = ext
                continue
            
            a = curr - (rLen & (r - 1))
            if a == curr:
                continue
            
            ext = min(a + r, b)
            self.bSort(arr, a, ext, curr)
            a = ext
        
        return mono
    
    def smartCollect(self, arr, a, b, q, r):
        kA = b - 1 # Key start
        k = 1 # Immediately absorb first value as key
        
        rP = b - 2
        nxt = kA - ((kA - a) & (r - 1)) # Initialise next run start as start of last run
        
        while nxt >= a and k < q: # Search until start is reached or keys meet target
            kP = kA + k - 1
            while rP >= nxt and k < q:
                kP = self.expSearchBW(arr, arr[rP], kA, kP + 1, False) - 1
                if kP < kA or arr[kP] != arr[rP]:
                    self.rotate(arr, rP + 1, kA, kA + k)
                    ofs = kA - rP - 1
                    kP -= ofs
                    self.insertRight(arr, rP, kP)
                    kA -= ofs + 1
                    k += 1
                rP = self.expSearchBW(arr, arr[kP], nxt, rP, True) - 1
            
            if k >= r:
                for left in range(a, nxt, 2 * r):
                    mid = left + r
                    right = min(left + 2 * r, nxt)
                    
                    if mid >= right:
                        break
                    
                    ls, rs = self.shrinkBounds(arr, left, mid, right)
                    if ls < 0:
                        continue
                    
                    if mid - ls <= rs - mid:
                        self.mergeFW(arr, ls, mid, rs, kA)
                    else:
                        self.mergeBW(arr, ls, mid, rs, kA)
                
                self.shellsort(arr, kA, kA + self.lastUsed)
                self.lastUsed = 0
                r *= 2
            
            nxt -= ((nxt - a - 1) & (r - 1)) + 1
        
        self.rotate(arr, kA, kA + k, b)
        
        if k < q:
            k = self.floorPow2(k)
            if k <= self.smallMerge:
                l = kA - ((kA - a) & (r - 1))
                self.buildRuns(arr, l, b, self.minRun)
                return 0
        
        l = kA - ((kA - a) & (r - 1))
        
        frag = 0
        prev = r
        
        while l < b - k:
            m = l + 1
            while m < b - k and arr[m - 1] <= arr[m]:
                m += 1
            
            if r > self.minRun and m - l <= r // 2 and prev + m - l <= r:
                r //= 2
                frag &= (r - 1)
            
            prev = m - l
            ls, rs = self.shrinkBounds(arr, l - frag, l, l + min(r - frag, prev))
            if ls >= 0:
                if l - ls <= rs - l:
                    self.mergeFW(arr, ls, l, rs, b - k)
                else:
                    self.mergeBW(arr, ls, l, rs, b - k)
            frag = (frag + prev) & (r - 1)
            l = m
        
        while r <= k:
            for left in range(a, b - k, 2 * r):
                mid = left + r
                right = min(left + 2 * r, b - k)
                
                if mid >= right:
                    break
                
                ls, rs = self.shrinkBounds(arr, left, mid, right)
                if ls < 0:
                    continue
                
                if mid - ls <= rs - mid:
                    self.mergeFW(arr, ls, mid, rs, b - k)
                else:
                    self.mergeBW(arr, ls, mid, rs, b - k)
            r *= 2
        
        self.shellsort(arr, b - k, b - k + self.lastUsed)
        self.lastUsed = 0
        
        return k
    
    def redistBuffer(self, arr, a, m, b):
        L = m - a
        R = b - m
        while min(L, R) > self.smallMerge:
            L = m - a
            R = b - m
            
            fw = (L <= R)
            if fw:
                if arr[a] > arr[m + L - 1]:
                    p = self.expSearchFW(arr, arr[a], m + L - 1, b, True)
                    self.rotate(arr, a, m, p)
                    d = p - m
                    a += d + 1
                    m += d
                    L -= 1
                    continue
            else:
                if arr[b - 1] < arr[m - R]:
                    p = self.expSearchBW(arr, arr[b - 1], a, m - R, False)
                    self.rotate(arr, p, m, b)
                    d = m - p
                    b -= d + 1
                    m -= d
                    R -= 1
                    continue
            
            lo = 0
            hi = min(L, R)
            while lo < hi:
                q = (lo + hi) // 2
                if arr[m - 1 - q] > arr[m + q]:
                    lo = q + 1
                else:
                    hi = q
            d = lo
            
            self.blockSwap(arr, m - d, m, d)
            
            if fw:
                self.lazyMerge(arr, a, m - d, m, True)
                a = m
                m += d
            else:
                self.lazyMerge(arr, m, m + d, b, True)
                b = m
                m -= d
        self.lazyMerge(arr, a, m, b, True)
    
    def lazyStableSort(self, arr, a, b, built):
        if not built and self.buildRuns(arr, a, b, self.minRun):
            return
        
        n = b - a
        
        N = self.minRun
        while N < n:
            for left in range(a, b, 2 * N):
                mid = left + N
                right = min(left + 2 * N, b)
                
                if mid >= right:
                    break
                
                ls, rs = self.shrinkBounds(arr, left, mid, right)
                if ls < 0:
                    continue
                
                self.redistBuffer(arr, ls, mid, rs) # maybe this is faster
            N *= 2
    
    def sortInPlace(self, arr, a, b):
        n = b - a
        
        if n <= 2 * self.minRun:
            h = self.countRun(arr, a, b)
            self.bSort(arr, a, b, h)
            return
        
        if n <= (self.minRun * self.minRun) // 2:
            self.lazyStableSort(arr, a, b, False)
            return
        
        if self.buildRuns(arr, a, b, self.minRun):
            return
        
        small = n <= self.minRun * self.minRun
        
        bLen = self.sqrtPow2(n)
        
        if small:
            target = n // bLen
        else:
            target = bLen + n // bLen
        
        keys = self.smartCollect(arr, a, b, target, self.minRun)
        b1 = b - keys
        
        if keys == 0:
            self.lazyStableSort(arr, a, b, True)
            return
        
        recalc = keys < target
        
        tLen = (n - keys) // bLen
        t = b1
        buf = b1 + tLen
        bufLen = 0 if small else bLen
        
        N = max(self.minRun, 2 * self.floorPow2(keys))
        
        while N < n:
            if recalc:
                tLen1 = tLen
                root = self.sqrtPow2(2 * N)
                
                while 2 * root + (2 * N) // (2 * root) < keys:
                    root *= 2
                
                tLen = (2 * N) // root
                if tLen <= keys:
                    buf = b1 + tLen
                    bufLen = keys - tLen
                else:
                    while (2 * N) // root > keys:
                        root *= 2
                    bufLen = 0
                    tLen = (2 * N) // root
                bLen = root
                self.bSort(arr, t, t + tLen, t + tLen1)
            
            for left in range(a, b1, N * 2):
                mid = min(left + N, b1)
                right = min(left + 2 * N, b1)
                
                if mid >= right:
                    break
                
                ls, rs = self.shrinkBounds(arr, left, mid, right)
                
                if ls < 0:
                    continue
                
                if mid - ls > bLen:
                    ls -= (ls - left) & (bLen - 1)
                
                if min(rs - mid, mid - ls) <= self.smallMerge:
                    self.lazyMerge(arr, ls, mid, rs, True)
                elif mid - ls <= bufLen:
                    self.mergeFW(arr, ls, mid, rs, buf)
                elif rs - mid <= bufLen:
                    self.mergeBW(arr, ls, mid, rs, buf)
                elif bLen <= bufLen:
                    self.blockMerge(arr, ls, mid, rs, bLen, buf, t)
                else:
                    self.blockMergeNoBuf(arr, ls, mid, rs, bLen, t)
            
            N *= 2
        
        if recalc:
            self.bSort(arr, t, b, t + tLen)
        else:
            self.shellsort(arr, buf, buf + self.lastUsed)
        self.redistBuffer(arr, a, b1, b)
    
    def arrCopy(self, frm, a, to, b, n):
        for i in range(n):
            to[b + i] = frm[a + i]
    
    def safeCopy(self, arr, a, b, n):
        if a > b:
            for i in range(n):
                arr[b + i] = arr[a + i]
        else:
            for i in range(n - 1, -1, -1):
                arr[b + i] = arr[a + i]
    
    def mergeFWAux(self, arr, a, m, b, buf, c): # Assumes already in buffer
        L = m - a
        i = 0
        while i < L and m < b:
            if buf[c + i] <= arr[m]:
                arr[a] = buf[c + i]
                i += 1
            else:
                arr[a] = arr[m]
                m += 1
            a += 1
        self.arrCopy(buf, i, arr, a, L - i)
    
    def mergeBWAux(self, arr, a, m, b, buf, c): # Assumes already in buffer
        R = b - m
        b -= 1; m -= 1
        while R > 0 and m >= a:
            if buf[c + R - 1] >= arr[m]:
                arr[b] = buf[c + R - 1]
                R -= 1
            else:
                arr[b] = arr[m]
                m -= 1
            b -= 1
        self.arrCopy(buf, c, arr, b - R + 1, R)
    
    def mergeAux(self, arr, a, m, b, buf):
        a, b = self.shrinkBounds(arr, a, m, b)
        if a < 0:
            return
        L = m - a
        R = b - m
        if L <= R:
            self.arrCopy(arr, a, buf, 0, L)
            self.mergeFWAux(arr, a, m, b, buf, 0)
        else:
            self.arrCopy(arr, m, buf, 0, R)
            self.mergeBWAux(arr, a, m, b, buf, 0)
    
    def mergeTo(self, arr, a, L, R, buf, b): # Lengths for convenience
        i = a + L
        while L > 0 and R > 0:
            if arr[a] <= arr[i]:
                buf[b] = arr[a]
                a += 1
                L -= 1
            else:
                buf[b] = arr[i]
                i += 1
                R -= 1
            b += 1
        self.arrCopy(arr, a, buf, b, L)
        b += L
        self.arrCopy(arr, i, buf, b, R)
    
    def mergeFour(self, arr, a, W, X, Y, Z, buf):
        b = a + W
        c = b + X
        d = c + Y
        e = d + Z
        
        midL = a + W // 2
        midR = d + Z // 2
        shrinkL = arr[midL] <= arr[b]
        shrinkR = arr[d - 1] <= arr[midR]
        
        if shrinkL and shrinkR:
            self.mergeAux(arr, midL, b, c, buf)
            self.mergeAux(arr, c, d, e, buf)
            self.mergeAux(arr, a, c, e, buf)
            return
        
        if shrinkL:
            self.mergeAux(arr, midL, b, c, buf)
            self.mergeTo(arr, c, Y, Z, buf, 0)
            self.mergeBWAux(arr, a, c, e, buf, 0)
            return
        
        if shrinkR:
            self.mergeAux(arr, c, d, e, buf)
            self.mergeTo(arr, a, W, X, buf, 0)
            self.mergeFWAux(arr, a, c, e, buf, 0)
            return
        
        self.mergeTo(arr, a, W, X, buf, 0)
        self.mergeTo(arr, c, Y, Z, buf, W + X)
        self.mergeTo(buf, 0, W + X, Y + Z, arr, a)
    
    def scrollMergeAux(self, arr, p, a, m, left):
        i = m
        if left:
            while a < m:
                if arr[a] <= arr[i]:
                    arr[p] = arr[a]
                    a += 1
                else:
                    arr[p] = arr[i]
                    i += 1
                p += 1
        else:
            while a < m:
                if arr[a] < arr[i]:
                    arr[p] = arr[a]
                    a += 1
                else:
                    arr[p] = arr[i]
                    i += 1
                p += 1
        return i
    
    def tailMergeAux(self, arr, p, a, m, b, buf, bLen):
        i = m
        
        while a < m and i < b:
            if arr[a] <= arr[i]:
                arr[p] = arr[a]
                a += 1
            else:
                arr[p] = arr[i]
                i += 1
            p += 1
        
        if a < m:
            if a > p:
                self.safeCopy(arr, a, p, m - a)
            self.arrCopy(buf, 0, arr, b - bLen, bLen)
            return
        
        a = 0
        while a < bLen and i < b:
            if buf[a] <= arr[i]:
                arr[p] = buf[a]
                a += 1
            else:
                arr[p] = arr[i]
                i += 1
            p += 1
        
        self.arrCopy(buf, a, arr, p, bLen - a)
    
    def blockCycle(self, arr, a, lCount, rCount, bLen, tags, buf):
        tags[0] = (lCount - 1) << 1
        l = 0
        m = lCount
        r = m
        b = m + rCount
        o = 1
        while l < m - 1 and r < b:
            if arr[a + (l + 1) * bLen - 1] <= arr[a + (r + 1) * bLen - 1]:
                tags[o] = l << 1
                l += 1
            else:
                tags[o] = (r << 1) | 1
                r += 1
            o += 1
        while l < m - 1:
            tags[o] = l << 1
            l += 1
            o += 1
        while r < b:
            tags[o] = (r << 1) | 1
            r += 1
            o += 1
        
        total = lCount + rCount
        for i in range(total):
            if (tags[i] >> 1) != i:
                self.arrCopy(arr, a + i * bLen, buf, 0, bLen)
                j = i
                nxt = (tags[i] >> 1)
                while nxt != i:
                    self.safeCopy(arr, a + nxt * bLen, a + j * bLen, bLen)
                    tags[j] = (j << 1) | (tags[j] & 1)
                    j = nxt
                    nxt = (tags[nxt] >> 1)
                self.arrCopy(buf, 0, arr, a + j * bLen, bLen)
                tags[j] = (j << 1) | (tags[j] & 1)
    
    def blockMergeAux(self, arr, a, m, b, bLen, tags, buf):
        leftLen = m - a
        rightLen = b - m
        lCount = leftLen // bLen
        rCount = rightLen // bLen
        bCount = lCount + rCount
        rem = rightLen & (bLen - 1)
        
        self.blockCycle(arr, a, lCount, rCount, bLen, tags, buf)
        self.arrCopy(arr, a, buf, 0, bLen)
        
        f = a + bLen
        left = (tags[1] & 1) == 0
        for i in range(2, bCount):
            if left ^ ((tags[i] & 1) == 0):
                nxt = a + i * bLen
                f = self.scrollMergeAux(arr, f - bLen, f, nxt, left)
                left = not left
        
        if left:
            self.tailMergeAux(arr, f - bLen, f, b - rem, b, buf, bLen)
        else:
            self.tailMergeAux(arr, f - bLen, f, f, b, buf, bLen)
    
    def sortAux(self, arr, a, b, mem):
        n = b - a
        
        if n < 2 * self.minRun:
            hint = self.countRun(arr, a, b)
            self.bSort(arr, a, b, hint)
            return
        
        if n <= (self.minRun * self.minRun) // 2:
            self.lazyStableSort(arr, a, b, False)
            return
        
        if self.buildRuns(arr, 0, n, self.minRun):
            return
        
        if mem < n // 2:
            bLen = self.sqrtPow2(n)
            
            if mem < bLen: # If memory is insufficient, return to sorting in-place
                self.sortInPlace(arr, a, b)
                return
            
            while bLen < mem:
                bLen *= 2
            
            tLen = n // bLen
            
            buf = [0] * bLen
            tags = [0] * tLen
        else:
            bLen = min(mem, n)
            if bLen < n:
                bLen = n // 2
            buf = [0] * bLen
        
        N = self.minRun
        
        while N * 4 <= bLen:
            i = a
            while i + 4 * N <= b:
                self.mergeFour(arr, i, N, N, N, N, buf)
                i += 4 * N
            if i + 3 * N < b:
                self.mergeFour(arr, i, N, N, N, (b - a) & (N - 1), buf)
            elif i + 2 * N < b:
                mergeL = (arr[i + N - 1] > arr[i + N])
                if mergeL:
                    self.mergeTo(arr, i, N, N, buf, 0)
                    self.mergeFWAux(arr, i, i + 2 * N, b, buf, 0)
                else:
                    self.mergeAux(arr, i, i + 2 * N, b, buf)
            elif i + N < b:
                self.mergeAux(arr, i, i + N, b, buf)
            N *= 4
        
        while N < n:
            for left in range(a, b, 2 * N):
                mid = left + N
                right = min(left + N * 2, b)
                
                if mid >= right:
                    break
                
                l, r = self.shrinkBounds(arr, left, mid, right)
                if l < 0:
                    continue
                
                lenL = mid - l
                lenR = r - mid
                
                if min(lenL, lenR) > bLen:
                    l -= (l - left) & (bLen - 1)
                
                if min(lenL, lenR) <= bLen:
                    self.mergeAux(arr, l, mid, r, buf)
                else:
                    self.blockMergeAux(arr, l, mid, r, bLen, tags, buf)
            N *= 2
