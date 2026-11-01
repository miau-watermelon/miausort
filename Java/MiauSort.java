/*
MIT License

Copyright (c) 2013 Andrey Astrelin
Copyright (c) 2020 Amari Calipso
Copyright (c) 2026 bzyjin
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
*/

package miausort;

public class MiauSort<T extends Comparable<? super T>> {
    private static final int MIN_MERGE = 32;
    private static final int MIN_GALLOP = 7;

    private void swap(T[] array, int a, int b) {
        T tmp = array[a];
        array[a] = array[b];
        array[b] = tmp;
    }

    private void reverse(T[] array, int a, int b) {
        --b;
        while (a < b)
            swap(array, a++, b--);
    }

    private void blockSwap(T[] array, int a, int b, int length) {
        while (length-- > 0)
            swap(array, a++, b++);
    }

    private void insertLeft(T[] array, int a, int b) {
    	if (a <= b) return;
        T tmp = array[a];
        switch (a-b) {
        	case 2: array[b+2] = array[b+1];
        	case 1: array[b+1] = array[b];
    				break;
        	default: System.arraycopy(array, b, array, b + 1, a - b);
        }
        array[b] = tmp;
    }

    private void rotateAux(T[] array, int base, int leftLen, int rightLen, T[] buffer) {
        if (leftLen < 1 || rightLen < 1)
            return;

        if (leftLen == rightLen) {
            blockSwap(array, base, base + leftLen, rightLen);
            return;
        }
        
        int a = base,
            b = base + leftLen,
            c = base + rightLen,
            d = base + leftLen + rightLen;
        
        if (leftLen < rightLen) {
        	int bridge = rightLen - leftLen;
        	if (bridge < leftLen) {
        		if (bridge > buffer.length) {
        			rotateInPlace(array, base, leftLen, rightLen);
        			return;
        		}
        		System.arraycopy(array, b, buffer, 0, bridge);
        		while (leftLen-- > 0) {
        			array[--c] = array[--d];
        			array[d] = array[--b];
        		}
        		System.arraycopy(buffer, 0, array, a, bridge);
        	} else {
        		if (leftLen > buffer.length) {
        			rotateInPlace(array, base, leftLen, rightLen);
        			return;
        		}
        		System.arraycopy(array, a, buffer, 0, leftLen);
        		System.arraycopy(array, b, array, a, rightLen);
        		System.arraycopy(buffer, 0, array, c, leftLen);
        	}
        } else {
        	int bridge = leftLen - rightLen;
        	if (bridge < rightLen) {
        		if (bridge > buffer.length) {
        			rotateInPlace(array, base, leftLen, rightLen);
        			return;
        		}
        		System.arraycopy(array, c, buffer, 0, bridge);
        		while (rightLen-- > 0) {
        			array[c++] = array[a];
        			array[a++] = array[b++];
        		}
        		System.arraycopy(buffer, 0, array, d - bridge, bridge);
        	} else {
        		if (rightLen > buffer.length) {
        			rotateInPlace(array, base, leftLen, rightLen);
        			return;
        		}
        		System.arraycopy(array, b, buffer, 0, rightLen);
        		System.arraycopy(array, b - leftLen, array, d - leftLen, leftLen);
        		System.arraycopy(buffer, 0, array, a, rightLen);
        	}
        }
    }

    private void rotateInPlace(T[] array, int base, int leftLen, int rightLen) {
        if (leftLen < 1 || rightLen < 1)
            return;

        int a = base,
            b = base + leftLen - 1,
            c = base + leftLen,
            d = base + leftLen + rightLen - 1;
        
        while (a < b && c < d) {
            T tmp = array[b];
            array[b--] = array[a];
            array[a++] = array[c];
            array[c++] = array[d];
            array[d--] = tmp;
        }
        while (a < b) {
            T tmp = array[b];
            array[b--] = array[a];
            array[a++] = array[d];
            array[d--] = tmp;
        }
        while (c < d) {
            T tmp = array[c];
            array[c++] = array[d];
            array[d--] = array[a];
            array[a++] = tmp;
        }
        reverse(array, a, d + 1);
    }

    private int binarySearch(T[] array, int base, int length, T target, boolean left) {
    	if (length < 1) return base;
        while (length > 0) {
            int mid = length >> 1;
            int cmp = array[base + mid].compareTo(target);
            if (cmp < 0 || (!left && cmp == 0)) {
                base += mid + 1;
                length -= mid + 1;
            } else {
                length = mid;
            }
        }
        return base;
    }

    private int expSearchFW(T[] array, int base, int length, T target, boolean left) {
        if (length < 1) return base;
        int tieBreak = left ? 0 : 1;
        int ofs = 1,
            lastOfs = 0;
        while (ofs < length && array[base + ofs].compareTo(target) < tieBreak) {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if (ofs <= 0) ofs = length;
        }
        if (ofs > length) ofs = length;
        return binarySearch(array, base + lastOfs, ofs - lastOfs, target, left);
    }

    private int expSearchBW(T[] array, int base, int length, T target, boolean left) {
        if (length <= 0) return base;
        int tieBreak = left ? -1 : 0;
        int end = base + length - 1;
        int ofs = 1,
            lastOfs = 0;
        while (ofs < length && array[end - ofs].compareTo(target) > tieBreak) {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if (ofs <= 0) ofs = length;
        }
        if (ofs > length) ofs = length;
        return binarySearch(array, end - ofs + 1, ofs - lastOfs, target, left);
    }

    private int[] shrinkBounds(T[] array, int base, int leftLen, int rightLen, T[] buffer) {
    	if (leftLen < 1 || rightLen < 1) return new int[] {-1, 0, 0};
        int mid = base + leftLen;
        if (array[mid - 1].compareTo(array[mid]) <= 0)
            return new int[] {-1, 0, 0};

        int start = expSearchFW(array, base, leftLen, array[mid], false);
        int end = expSearchBW(array, mid, rightLen, array[mid - 1], true);
        
        if (array[start].compareTo(array[end - 1]) > 0) {
            rotateAux(array, start, mid - start, end - mid, buffer);
            return new int[] {-1, 0, 0};
        }
        return new int[] {start, mid - start, end - mid};
    }

    private int countRun(T[] array, int base, int length) {
        int runLen = 1;
        while (runLen < length && array[base + runLen - 1].compareTo(array[base + runLen]) <= 0)
            runLen++;
        if (runLen >= length || runLen > 1 && array[base].compareTo(array[base + runLen - 1]) < 0)
            return runLen;
        reverse(array, base, base + runLen);
        T prev = array[base];
        int segment = runLen;
        while (segment < length && array[base + segment].compareTo(prev) < 0) {
            runLen++;
            while (runLen < length && array[base + runLen].compareTo(array[base + runLen - 1]) == 0)
                runLen++;
            reverse(array, base + segment, base + runLen);
            prev = array[base + runLen - 1];
            segment = runLen;
        }
        reverse(array, base, base + runLen);
        return runLen;
    }

    private void binarySort(T[] array, int base, int length, int hint) {
        while (hint < length) {
            if (array[base + hint - 1].compareTo(array[base + hint]) > 0)
                insertLeft(array, base + hint, binarySearch(array, base, hint, array[base + hint], false));
            hint++;
        }
    }

    private boolean buildRuns(T[] array, int base, int length, int minRun) {
        boolean mono = true;
        while (length > 0) {
            int runLen = countRun(array, base, length);
            if (runLen == length)
                break;

            mono = false;

            if (runLen < minRun) {
                int ext = minRun < length ? minRun : length;
                binarySort(array, base, ext, runLen);
                base += ext;
                length -= ext;
                continue;
            }

            int rem = runLen % minRun;
            base += runLen - rem;
            length -= runLen - rem;
            if (rem == 0)
                continue;
            runLen = rem;

            int ext = minRun < length ? minRun : length;
            binarySort(array, base, ext, runLen);
            base += ext;
            length -= ext;
        }
        return mono;
    }

    private void mergeFromFW(T[] array, int base, int leftLen, int rightLen, T[] buffer, int bufBase) {
    	if (leftLen < 1) return;
    	if (rightLen < 1) {
    		System.arraycopy(buffer, bufBase, array, base, leftLen);
    		return;
    	}
    	int mid = base + leftLen;
    	int t = Math.max(MIN_GALLOP, 31 - Integer.numberOfLeadingZeros(leftLen + rightLen));
    	
    	outer: while (true) {
    		int c1 = 0,
				c2 = 0;
    		
    		do {
    			if (buffer[bufBase].compareTo(array[mid]) <= 0) {
    				array[base++] = buffer[bufBase++];
    				++c1;
    				c2 = 0;
    				if (--leftLen < 1) break outer;
    			} else {
    				array[base++] = array[mid++];
    				++c2;
    				c1 = 0;
    				if (--rightLen < 1) break outer;
    			}
    		} while ((c1 | c2) <= t);
    		
    		do {
    			c1 = expSearchFW(buffer, bufBase, leftLen, array[mid], false) - bufBase;
    			if (c1 > 0) {
    				System.arraycopy(buffer, bufBase, array, base, c1);
    				bufBase += c1;
    				base += c1;
    				leftLen -= c1;
    				if (leftLen < 1) break outer;
    			}
    			
    			array[base++] = array[mid++];
    			if (--rightLen < 1) break outer;
    			
    			c2 = expSearchFW(array, mid, rightLen, buffer[bufBase], true) - mid;
    			if (c2 > 0) {
    				System.arraycopy(array, mid, array, base, c2);
    				mid += c2;
    				base += c2;
    				rightLen -= c2;
    				if (rightLen < 1) break outer;
    			}
    			
    			array[base++] = buffer[bufBase++];
    			if (--leftLen < 1) break outer;
    		} while (c1 > t || c2 > t);
    	}
    	if (leftLen > 0) System.arraycopy(buffer, bufBase, array, base, leftLen);
    }

    private void mergeFromBW(T[] array, int base, int leftLen, int rightLen, T[] buffer, int bufBase, boolean left) {
    	if (rightLen < 1) return;
    	if (leftLen < 1) {
    		System.arraycopy(buffer, bufBase, array, base, rightLen);
    		return;
    	}
    	
    	int mid = base + leftLen - 1,
    		end = base + leftLen + rightLen - 1,
    		bufEnd = bufBase + rightLen - 1;
    	
    	int t = Math.max(MIN_GALLOP, 31 - Integer.numberOfLeadingZeros(leftLen + rightLen));
    	
    	outer: while (true) {
    		int c1 = 0,
				c2 = 0;
    		
    		do {
    			int cmp = buffer[bufEnd].compareTo(array[mid]);
    			if (cmp > 0 || (left && cmp == 0)) {
    				array[end--] = buffer[bufEnd--];
    				++c2;
    				c1 = 0;
    				if (--rightLen < 1) break outer;
    			} else {
    				array[end--] = array[mid--];
    				++c1;
    				c2 = 0;
    				if (--leftLen < 1) break outer;
    			}
    		} while ((c1 | c2) <= t);
    		
    		do {
    			c2 = bufEnd+1 - expSearchBW(buffer, bufBase, rightLen, array[mid], left);
    			if (c2 > 0) {
    				bufEnd -= c2;
    				rightLen -= c2;
    				end -= c2;
    				System.arraycopy(buffer, bufEnd+1, array, end+1, c2);
    				if (rightLen < 1) break outer;
    			}
    			
    			array[end--] = array[mid--];
    			if (--leftLen < 1) break outer;
    			
    			c1 = mid+1 - expSearchBW(array, base, leftLen, buffer[bufEnd], !left);
    			if (c1 > 0) {
    				mid -= c1;
    				leftLen -= c1;
    				end -= c1;
    				System.arraycopy(array, mid+1, array, end+1, c1);
    				if (leftLen < 1) break outer; 
    			}
    			
    			array[end--] = buffer[bufEnd--];
    			if (--rightLen < 1) break outer;
    		} while (c1 > t || c2 > t);
    	}
    	if (rightLen > 0) System.arraycopy(buffer, bufBase, array, end-rightLen+1, rightLen);
    }

    private void merge(T[] array, int base, int leftLen, int rightLen, T[] buffer) {
        int[] bnd = shrinkBounds(array, base, leftLen, rightLen, buffer);
        if (bnd[0] < 0)
            return;
        base = bnd[0];
        leftLen = bnd[1];
        rightLen = bnd[2];
        if (leftLen <= rightLen) {
            System.arraycopy(array, base, buffer, 0, leftLen);
            mergeFromFW(array, base, leftLen, rightLen, buffer, 0);
        } else {
            System.arraycopy(array, base + leftLen, buffer, 0, rightLen);
            mergeFromBW(array, base, leftLen, rightLen, buffer, 0, true);
        }
    }

    private void mergeTo(T[] array, int base, int leftLen, int rightLen, T[] buffer, int bufBase) {
        int mid = base + leftLen;
        
        int t = Math.max(MIN_GALLOP, 31 - Integer.numberOfLeadingZeros(leftLen + rightLen));
        
        outer: while (true) {
        	int c1 = 0,
    			c2 = 0;
        	do {
	        	if (array[base].compareTo(array[mid]) <= 0) {
	        		buffer[bufBase++] = array[base++];
	        		++c1;
	        		c2 = 0;
	        		if (--leftLen < 1) break outer;
	        	} else {
	        		buffer[bufBase++] = array[mid++];
	        		++c2;
	        		c1 = 0;
	        		if (--rightLen < 1) break outer;
	        	}
        	} while ((c1 | c2) <= t);
        	
        	do {
        		c1 = expSearchFW(array, base, leftLen, array[mid], false) - base;
        		if (c1 > 0) {
        			System.arraycopy(array, base, buffer, bufBase, c1);
        			base += c1;
        			bufBase += c1;
        			leftLen -= c1;
        			if (leftLen < 1) break outer;
        		}
        		
        		buffer[bufBase++] = array[mid++];
        		if (--rightLen < 1) break outer;
        		
        		c2 = expSearchFW(array, mid, rightLen, array[base], true) - mid;
        		if (c2 > 0) {
        			System.arraycopy(array, mid, buffer, bufBase, c2);
        			mid += c2;
        			bufBase += c2;
        			rightLen -= c2;
        			if (rightLen < 1) break outer;
        		}
        		
        		buffer[bufBase++] = array[base++];
        		if (--leftLen < 1) break outer;
        	} while (c1 > t || c2 > t);
        }
        if (leftLen > 0) System.arraycopy(array, base, buffer, bufBase, leftLen);
        else System.arraycopy(array, mid, buffer, bufBase, rightLen);
    }

    private void mergeFour(T[] array, int base, int W, int X, int Y, int Z, T[] buffer) {
    	assert W+X >= Y+Z: "Left subarray pair must be larger than or equal to right pair!";
        int base1 = base + W + X;
        
        boolean skipL = array[base+W-1].compareTo(array[base+W]) <= 0,
        		skipR = array[base1+Y-1].compareTo(array[base1+Y]) <= 0;
        
        if (skipL && skipR) {
        	merge(array, base, W+X, Y+Z, buffer);
        	return;
        }
        
        if (skipL) {
        	mergeTo(array, base1, Y, Z, buffer, 0);
        	mergeFromBW(array, base, W+X, Y+Z, buffer, 0, true);
        	return;
        }
        
        if (skipR) {
        	mergeTo(array, base, W, X, buffer, 0);
        	mergeFromFW(array, base, W+X, Y+Z, buffer, 0);
        	return;
        }
        
        mergeTo(array, base, W, X, buffer, 0);
        mergeTo(array, base1, Y, Z, array, base);
        mergeFromBW(array, base, Y+Z, W+X, buffer, 0, false);
    }

    private void smallMergeSort(T[] array, int base, int length, int minRun) {
        if (buildRuns(array, base, length, minRun))
            return;

        int half = length / 2;
        @SuppressWarnings("unchecked")
        T[] buffer = (T[]) java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), half);
        int size = minRun;
        while (size <= half / 2) {
            int mergeOfs = 0;
            while (mergeOfs <= length - 4 * size) {
                mergeFour(array, base + mergeOfs, size, size, size, size, buffer);
                mergeOfs += 4 * size;
            }
            if (mergeOfs + 3 * size < length) {
                int remSize = length - (mergeOfs + 3 * size);
                mergeFour(array, base + mergeOfs, size, size, size, remSize, buffer);
            } else if (mergeOfs + 2 * size < length) {
                int remSize = length - (mergeOfs + 2 * size);
                mergeTo(array, base + mergeOfs + size, size, remSize, buffer, 0);
                mergeFromBW(array, base + mergeOfs, size, size + remSize, buffer, 0, true);
            } else if (mergeOfs + size < length) {
                int remSize = length - (mergeOfs + size);
                merge(array, base + mergeOfs, size, remSize, buffer);
            }
            size *= 4;
        }
        while (size < length) {
            int mergeOfs = 0;
            while (mergeOfs + 2 * size <= length) {
                merge(array, base + mergeOfs, size, size, buffer);
                mergeOfs += 2 * size;
            }
            if (mergeOfs < length - size) {
                int remSize = length - (mergeOfs + size);
                merge(array, base + mergeOfs, size, remSize, buffer);
            }
            size *= 2;
        }
    }

    private int scrollMerge(T[] array, int hole, int start, int mid, int end, boolean left) {
        int t = Math.max(MIN_GALLOP, 31 - Integer.numberOfLeadingZeros(end - start));
        int i = mid;
        while (true) {
        	int c1 = 0,
    			c2 = 0;
        	do {
        		int cmp = array[start].compareTo(array[i]);
        		if (cmp < 0 || (left && cmp == 0)) {
        			array[hole++] = array[start++];
        			++c1;
        			c2 = 0;
        			if (start >= mid) return i;
        		} else {
        			array[hole++] = array[i++];
        			++c2;
        			c1 = 0;
        			if (i >= end) throw new IllegalArgumentException("Comparison method violates its general contract!");
        		}
        	} while ((c1 | c2) <= t);
        	
        	do {
        		c1 = expSearchFW(array, start, mid-start, array[i], !left) - start;
        		if (c1 > 0) {
        			System.arraycopy(array, start, array, hole, c1);
        			start += c1;
        			hole += c1;
        			if (start >= mid) return i;
        		}
        		
        		array[hole++] = array[i++];
        		if (i >= end) throw new IllegalArgumentException("Comparison method violates its general contract!");
        		
        		c2 = expSearchFW(array, i, end-i, array[start], left) - i;
        		if (c2 > 0) {
        			System.arraycopy(array, i, array, hole, c2);
        			i += c2;
        			hole += c2;
        			if (i >= end) throw new IllegalArgumentException("Comparison method violates its general contract!");
        		}
        		
        		array[hole++] = array[start++];
        		if (start >= mid) return i;
        	} while (c1 > t || c2 > t);
        }
    }

    private void tailMerge(T[] array, int hole, int start, int mid, int end, T[] buffer, int blockLen) {
    	int i = mid;
    	while (start < mid && i < end) {
    		if (array[start].compareTo(array[i]) <= 0) {
    			array[hole++] = array[start++];
    		} else {
    			array[hole++] = array[i++];
    		}
    	}
    	if (start < mid) {
    		if (start > hole)
    			System.arraycopy(array, start, array, hole, mid - start);
    		System.arraycopy(buffer, 0, array, end - blockLen, blockLen);
    		return;
    	}
    	
    	mergeFromFW(array, hole, blockLen, end-i, buffer, 0);
    }

    private void getIndices(T[] array, int base, int leftCount, int rightCount, int blockLen, int[] indices) {
        int left = leftCount - 1,
            mid = leftCount,
            right = leftCount,
            end = leftCount + rightCount,
            out = 0;
        while (right < end && array[base + (left + 1) * blockLen - 1].compareTo(array[base + (right + 1) * blockLen - 1]) > 0)
        	indices[out++] = ((right++) << 1) | 1;
        indices[out++] = left << 1;
        left = 0;
        while (left < mid - 1 && right < end) {
            if (array[base + (left + 1) * blockLen - 1].compareTo(array[base + (right + 1) * blockLen - 1]) <= 0) {
                indices[out++] = (left++) << 1;
            } else {
                indices[out++] = ((right++) << 1) | 1;
            }
        }
        while (left < mid - 1) {
            indices[out++] = (left++) << 1;
        }
        while (right < end) {
            indices[out++] = ((right++) << 1) | 1;
        }
    }
    
    private void blockCycle(T[] array, int base, int leftCount, int rightCount, int blockLen, int[] indices, int buf) {
        int total = leftCount + rightCount;
        for (int i = 0; i < total; i++) {
            if (indices[i] >> 1 != i) {
                System.arraycopy(array, base + i * blockLen, array, buf, blockLen);
                int j = i;
                int nxt = (indices[i] >> 1);
                do {
                    System.arraycopy(array, base + nxt * blockLen, array, base + j * blockLen, blockLen);
                    indices[j] = (j << 1) | (indices[j] & 1);
                    j = nxt;
                    nxt = (indices[nxt] >> 1);
                } while (nxt != i);
                System.arraycopy(array, buf, array, base + j * blockLen, blockLen);
                indices[j] = (j << 1) | (indices[j] & 1);
            }
        }
    }

    private void blockMerge(T[] array, int base, int leftLen, int rightLen, int blockLen, int[] indices, T[] buffer) {
    	assert leftLen > blockLen && rightLen > blockLen: "Subarrays are too small for block merging!";
        assert leftLen % blockLen == 0: "Left subarray must be multiple of block length!";
        
        int end = base + leftLen + rightLen;
        int leftCount = leftLen / blockLen,
            rightCount = rightLen / blockLen;
        int blockCount = leftCount + rightCount,
            rem = rightLen - (rightCount * blockLen);
        
        System.arraycopy(array, base + (leftCount - 1) * blockLen, buffer, 0, blockLen);
        System.arraycopy(array, base, array, base + (leftCount - 1) * blockLen, blockLen);
        getIndices(array, base + blockLen, leftCount - 1, rightCount, blockLen, indices);
        blockCycle(array, base + blockLen, leftCount - 1, rightCount, blockLen, indices, base);

        int frag = base + blockLen;
        boolean left = ((indices[0] & 1) == 0);
        for (int i = 1; i < blockCount - 1; i++) {
            if (left ^ ((indices[i] & 1) == 0)) {
                int nxt = base + (i+1) * blockLen;
                frag = scrollMerge(array, frag - blockLen, frag, nxt, nxt + blockLen, left);
                left = !left;
            }
        }
        tailMerge(array, frag - blockLen, frag, left ? (end - rem) : frag, end, buffer, blockLen);
    }

    private void blockMergeDecide(T[] array, int base, int leftLen, int rightLen, int blockLen, int[] indices, T[] buffer) {
        int[] bnd = shrinkBounds(array, base, leftLen, rightLen, buffer);
        if (bnd[0] < 0)
            return;
        int mergeBase = bnd[0],
    		mergeLeft = bnd[1],
    		mergeRight = bnd[2];
    	
        if (mergeLeft <= mergeRight && mergeLeft <= blockLen) {
            System.arraycopy(array, mergeBase, buffer, 0, mergeLeft);
            mergeFromFW(array, mergeBase, mergeLeft, mergeRight, buffer, 0);
        } else if (mergeRight <= blockLen) {
            System.arraycopy(array, mergeBase + mergeLeft, buffer, 0, mergeRight);
            mergeFromBW(array, mergeBase, mergeLeft, mergeRight, buffer, 0, true);
        } else {
        	if (mergeBase > base) {
	        	int diff = (mergeBase - base) % blockLen;
	            mergeBase -= diff;
	            mergeLeft += diff;
        	}
            blockMerge(array, mergeBase, mergeLeft, mergeRight, blockLen, indices, buffer);
        }
    }

    public void sort(T[] array, int base, int length) {
        if (array == null || length < 2)
            return;

        int minRun = length;
        while (minRun >= MIN_MERGE) {
            minRun = (minRun + 1) / 2;
        }

        if (length <= 2 * minRun) {
            int hint = countRun(array, base, length);
            binarySort(array, base, length, hint);
            return;
        }

        if (length <= minRun * minRun) {
            smallMergeSort(array, base, length, minRun);
            return;
        }

        if (buildRuns(array, base, length, minRun))
            return;

        int blockLen = minRun;
        while (blockLen * blockLen < length)
            blockLen *= 2;
        
        int blockCount = length / blockLen;
        
        @SuppressWarnings("unchecked")
        T[] buffer = (T[]) java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), blockLen);
        int[] indices = new int[blockCount];

        int size = minRun;

        while (size * 2 <= blockLen) {
            int mergeOfs = 0;
            while (mergeOfs + 4 * size <= length) {
                mergeFour(array, base + mergeOfs, size, size, size, size, buffer);
                mergeOfs += 4 * size;
            }
            if (mergeOfs + 3 * size < length) {
                int remSize = length - (mergeOfs + 3 * size);
                mergeFour(array, base + mergeOfs, size, size, size, remSize, buffer);
            } else if (mergeOfs + 2 * size < length) {
                int remSize = length - (mergeOfs + 2 * size);
                mergeTo(array, base + mergeOfs + size, size, remSize, buffer, 0);
                mergeFromBW(array, base + mergeOfs, size, size + remSize, buffer, 0, true);
            } else if (mergeOfs + size < length) {
                int remSize = length - (mergeOfs + size);
                merge(array, base + mergeOfs, size, remSize, buffer);
            }
            size *= 4;
        }

        while (size < length) {
            int mergeOfs = 0;
            while (mergeOfs + 2 * size <= length) {
                blockMergeDecide(array, base + mergeOfs, size, size, blockLen, indices, buffer);
                mergeOfs += 2 * size;
            }
            if (mergeOfs + size < length) {
                int remSize = length - (mergeOfs + size);
                blockMergeDecide(array, base + mergeOfs, size, remSize, blockLen, indices, buffer);
            }
            size *= 2;
        }
    }
    
    public void sort(T[] array) {
    	sort(array, 0, array.length);
    }
}
