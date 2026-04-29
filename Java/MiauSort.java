package miausort;

public class MiauSort<T extends Comparable<? super T>> {
    private final int MIN_MERGE = 32;
    private final int MIN_GALLOP = 7;

    private void swap(T[] array, int a, int b) {
        T tmp = array[a];
        array[a] = array[b];
        array[b] = tmp;
    }

    private void reverse(T[] array, int a, int b) {
        b--;
        while (a < b)
            swap(array, a++, b--);
    }

    private void blockSwap(T[] array, int a, int b, int length) {
        while (length-- > 0)
            swap(array, a++, b++);
    }

    private void insertLeft(T[] array, int a, int b) {
        T tmp = array[a];
        System.arraycopy(array, b, array, b + 1, a - b);
        array[b] = tmp;
    }

    private void rotateAux(T[] array, int base, int leftLen, int rightLen, T[] buffer) {
        if (leftLen < 1 || rightLen < 1)
            return;

        if (leftLen == rightLen) {
            blockSwap(array, base, base + leftLen, rightLen);
            return;
        }
        
        if ((leftLen < rightLen ? rightLen - leftLen : leftLen - rightLen) > buffer.length) {
            rotateInPlace(array, base, leftLen, rightLen);
            return;
        }

        int a = base,
            b = base + leftLen,
            c = base + rightLen,
            d = base + leftLen + rightLen;
        
        if (leftLen < rightLen) {
            int bridge = rightLen - leftLen;
            if (bridge < leftLen) {
                System.arraycopy(array, b, buffer, 0, bridge);
                while (leftLen-- > 0) {
                    array[--c] = array[d];
                    array[--d] = array[--b];
                }
                System.arraycopy(buffer, 0, array, a, bridge);
            } else {
                System.arraycopy(array, a, buffer, 0, leftLen);
                System.arraycopy(array, b, array, a, rightLen);
                System.arraycopy(buffer, 0, array, c, leftLen);
            }
        } else {
            int bridge = leftLen - rightLen;
            if (bridge > rightLen) {
                System.arraycopy(array, c, buffer, 0, bridge);
                while (rightLen-- > 0) {
                    array[c++] = array[a];
                    array[a++] = array[b++];
                }
                System.arraycopy(buffer, 0, array, d - bridge, bridge);
            } else {
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
        if (length <= 0)
            return base;
        int tieBreak = left ? 0 : 1;
        int ofs = 1,
            lastOfs = 0;
        while (ofs < length && array[base + ofs].compareTo(target) < tieBreak) {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if (ofs <= 0) // overflow
                ofs = length;
        }
        if (ofs > length) ofs = length;
        return binarySearch(array, base + lastOfs, ofs - lastOfs, target, left);
    }

    private int expSearchBW(T[] array, int base, int length, T target, boolean left) {
        if (length <= 0)
            return base;
        int tieBreak = left ? -1 : 0;
        int end = base + length - 1;
        int ofs = 1,
            lastOfs = 0;
        
        while (ofs < length && array[end - ofs].compareTo(target) > tieBreak) {
            lastOfs = ofs;
            ofs = (ofs << 1) + 1;
            if (ofs <= 0)
                ofs = length;
        }
        if (ofs > length) ofs = length;
        return binarySearch(array, end - ofs + 1, ofs - lastOfs, target, left);
    }

    private int[] shrinkBounds(T[] array, int base, int leftLen, int rightLen, T[] buffer) {
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
        int mid = base + leftLen;
        int i = 0, j = 0;
        while (i < leftLen && j < rightLen) {
            if (buffer[bufBase + i].compareTo(array[mid + j]) <= 0) {
                array[base++] = buffer[bufBase + (i++)];
            } else {
                array[base++] = array[mid + (j++)];
            }
        }
        System.arraycopy(buffer, bufBase + i, array, base, leftLen - i);
    }

    private void mergeFromBW(T[] array, int base, int leftLen, int rightLen, T[] buffer, int bufBase) {
        int mid = base + leftLen - 1,
            end = base + leftLen + rightLen - 1,
            bufEnd = bufBase + rightLen - 1;
        int i = 0, j = 0;
        while (i < leftLen && j < rightLen) {
            if (buffer[bufEnd - j].compareTo(array[mid - i]) >= 0) {
                array[end--] = buffer[bufEnd - (j++)];
            } else {
                array[end--] = array[mid - (i++)];
            }
        }
        System.arraycopy(buffer, bufBase, array, end - (rightLen - j) + 1, rightLen - j);
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
            mergeFromBW(array, base, leftLen, rightLen, buffer, 0);
        }
    }

    private void mergeTo(T[] from, int baseFrom, int leftLen, int rightLen, T[] to, int baseTo) {
        int mid = baseFrom + leftLen;
        while (leftLen > 0 && rightLen > 0) {
            if (from[baseFrom].compareTo(from[mid]) <= 0) {
                to[baseTo++] = from[baseFrom++];
                leftLen--;
            } else {
                to[baseTo++] = from[mid++];
                rightLen--;
            }
        }
        System.arraycopy(from, baseFrom, to, baseTo, leftLen);
        baseTo += leftLen;
        System.arraycopy(from, mid, to, baseTo, rightLen);
    }

    private void mergeFour(T[] array, int base, int W, int X, int Y, int Z, T[] buffer) {
        int a = base;
        int b = a + W;
        int c = b + X;
        int d = c + Y;

        int midL = a + W / 2,
            midR = d + Z / 2;
        
        boolean shrinkL = (array[midL].compareTo(array[b]) <= 0),
                shrinkR = (array[d - 1].compareTo(array[midR]) <= 0);
        
        if (shrinkL && shrinkR) {
            merge(array, a, W, X, buffer);
            merge(array, c, Y, Z, buffer);
            merge(array, a, W + X, Y + Z, buffer);
            return;
        }

        if (shrinkL) {
            merge(array, a, W, X, buffer);
            mergeTo(array, c , Y, Z, buffer, 0);
            mergeFromBW(array, a, W + X, Y + Z, buffer, 0);
            return;
        }

        if (shrinkR) {
            merge(array, c, Y, Z, buffer);
            mergeTo(array, a, W, X, buffer, 0);
            mergeFromBW(array, a, W + X, Y + Z, buffer, 0);
            return;
        }

        mergeTo(array, a, W, X, buffer, 0);
        mergeTo(array, c, Y, Z, buffer, W + X);
        mergeTo(buffer, 0, W + X, Y + Z, array, a);
    }

    private void smallMergeSort(T[] array, int base, int length, int minRun) {
        if (buildRuns(array, base, length, minRun))
            return;

        int half = length / 2;
        @SuppressWarnings("unchecked")
        T[] buffer = (T[]) java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), half);
        int size = minRun;
        while (size * 4 <= half) {
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
                boolean mergeL = array[base + mergeOfs + size - 1].compareTo(array[base + mergeOfs + size]) > 0;
                if (mergeL) {
                    mergeTo(array, base + mergeOfs, size, size, buffer, 0);
                    mergeFromFW(array, base + mergeOfs, 2 * size, remSize, buffer, 0);
                } else {
                    merge(array, base + mergeOfs, 2 * size, remSize, buffer);
                }
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
            if (mergeOfs + size < length) {
                int remSize = length - (mergeOfs + size);
                merge(array, base + mergeOfs, size, remSize, buffer);
            }
            size *= 2;
        }
    }

    private int scrollMerge(T[] array, int hole, int start, int mid, boolean left) {
        int i = mid;
        while (start < mid) {
            int cmp = array[start].compareTo(array[i]);
            if (cmp < 0 || (left && cmp == 0)) {
                array[hole++] = array[start++];
            } else {
                array[hole++] = array[i++];
            }
        }
        return i;
    }

    private int scrollMergeGallop(T[] array, int hole, int start, int mid, boolean left) {
        int i = mid;
        while (true) {
            int count = expSearchFW(array, start, mid - start, array[i], !left);
            System.arraycopy(array, start, array, hole, count);
            start += count;
            hole += count;
            if (start >= mid)
                break;
            while (array[start].compareTo(array[i]) > (left ? 0 : -1)) {
                array[hole++] = array[i++];
            }
        }
        return i;
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

        int a = 0;
        if ((end - i) <= MIN_GALLOP * blockLen) {
            while (a < blockLen && i < end) {
                if (buffer[a].compareTo(array[i]) <= 0) {
                    array[hole++] = buffer[a++];
                } else {
                    array[hole++] = array[i++];
                }
            }
        } else {
            while (true) {
                while (a < blockLen && buffer[a].compareTo(array[i]) <= 0) {
                    array[hole++] = buffer[a++];
                }
                if (a >= blockLen)
                    break;
                int count = expSearchFW(array, i, end - i, buffer[a], true) - i;
                System.arraycopy(array, i, array, hole, count);
                i += count;
                hole += count;
                if (i >= end)
                    break;
            }
        }
        System.arraycopy(buffer, a, array, hole, blockLen - a);
    }

    private void blockCycle(T[] array, int base, int leftCount, int rightCount, int blockLen, int[] tags, T[] buffer) {
        tags[0] = (leftCount - 1) << 1;
        int left = 0,
            mid = leftCount,
            right = leftCount,
            end = leftCount + rightCount,
            out = 1;
        while (left < mid - 1 && right < end) {
            if (array[base + (left + 1) * blockLen - 1].compareTo(array[base + (right + 1) * blockLen - 1]) <= 0) {
                tags[out++] = (left++) << 1;
            } else {
                tags[out++] = ((right++) << 1) | 1;
            }
        }
        while (left < mid - 1) {
            tags[out++] = (left++) << 1;
        }
        while (right < end) {
            tags[out++] = ((right++) << 1) | 1;
        }
        int total = leftCount + rightCount;
        for (int i = 0; i < total; i++) {
            if (tags[i] >> 1 != i) {
                System.arraycopy(array, base + i * blockLen, buffer, 0, blockLen);
                int j = i;
                int nxt = (tags[i] >> 1);
                do {
                    System.arraycopy(array, base + nxt * blockLen, array, base + j * blockLen, blockLen);
                    tags[j] = (j << 1) | (tags[j] & 1);
                    j = nxt;
                    nxt = (tags[nxt] >> 1);
                } while (nxt != i);
                System.arraycopy(buffer, 0, array, base + j * blockLen, blockLen);
                tags[j] = (j << 1) | (tags[j] & 1);
            }
        }
    }

    private void blockMerge(T[] array, int base, int leftLen, int rightLen, int blockLen, int[] tags, T[] buffer) {
        assert leftLen % blockLen == 0: "no left remainders pls :3";
        int end = base + leftLen + rightLen;
        int leftCount = leftLen / blockLen,
            rightCount = rightLen / blockLen;
        int blockCount = leftCount + rightCount,
            rem = rightLen - (rightCount * blockLen);
        
        blockCycle(array, base, leftCount, rightCount, blockLen, tags, buffer);
        System.arraycopy(array, base, buffer, 0, blockLen);

        int frag = base + blockLen;
        boolean left = ((tags[1] & 1) == 0);
        for (int i = 2; i < blockCount; i++) {
            if (left ^ ((tags[i] & 1) == 0)) {
                int nxt = base + i * blockLen;
                if (nxt - frag <= MIN_GALLOP * blockLen) {
                    frag = scrollMerge(array, frag - blockLen, frag, nxt, left);
                } else {
                    frag = scrollMergeGallop(array, frag - blockLen, frag, nxt, left);
                }
                left = !left;
            }
        }
        tailMerge(array, frag - blockLen, frag, left ? end - rem : frag, end, buffer, blockLen);
    }

    private void blockMergeDecide(T[] array, int base, int leftLen, int rightLen, int blockLen, int[] tags, T[] buffer) {
        int[] bnd = shrinkBounds(array, base, leftLen, rightLen, buffer);
        if (bnd[0] < 0)
            return;
        int mergeBase = bnd[0];
        leftLen = bnd[1];
        rightLen = bnd[2];

        if (leftLen > blockLen) {
            int diff = (mergeBase - base) % blockLen;
            mergeBase -= diff;
            leftLen += diff;
        }

        if (leftLen <= rightLen && leftLen <= blockLen) {
            System.arraycopy(array, mergeBase, buffer, 0, leftLen);
            mergeFromFW(array, mergeBase, leftLen, rightLen, buffer, 0);
        } else if (rightLen <= blockLen) {
            System.arraycopy(array, mergeBase + leftLen, buffer, 0, rightLen);
            mergeFromBW(array, mergeBase, leftLen, rightLen, buffer, 0);
        } else {
            blockMerge(array, mergeBase, leftLen, rightLen, blockLen, tags, buffer);
        }
    }

    public void sort(T[] array, int base, int length) {
        if (array == null || length <= 1)
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

        @SuppressWarnings("unchecked")
        T[] buffer = (T[]) java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), blockLen);
        int[] tags = new int [length / blockLen];

        int size = minRun;

        while (size * 4 <= blockLen) {
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
                boolean mergeL = array[base + mergeOfs + size - 1].compareTo(array[base + mergeOfs + size]) > 0;
                if (mergeL) {
                    mergeTo(array, base + mergeOfs, size, size, buffer, 0);
                    mergeFromFW(array, base + mergeOfs, 2 * size, remSize, buffer, 0);
                } else {
                    merge(array, base + mergeOfs, 2 * size, remSize, buffer);
                }
            } else if (mergeOfs + size < length) {
                int remSize = length - (mergeOfs + size);
                merge(array, base + mergeOfs, size, remSize, buffer);
            }
            size *= 4;
        }

        while (size < length) {
            int mergeOfs = 0;
            while (mergeOfs + 2 * size <= length) {
                blockMergeDecide(array, base + mergeOfs, size, size, blockLen, tags, buffer);
                mergeOfs += 2 * size;
            }
            if (mergeOfs + size < length) {
                int remSize = length - (mergeOfs + size);
                blockMergeDecide(array, base + mergeOfs, size, remSize, blockLen, tags, buffer);
            }
            size *= 2;
        }
    }
}
