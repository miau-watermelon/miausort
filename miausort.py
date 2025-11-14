class MiauBlockMerge:
    MIN_RUN = 24
    min_gallop = 7
    last_used = 0

    def __init__(self, memory_size):
        self.memory_size = int(memory_size)

    def swap(self, arr, a, b):
        arr[a], arr[b] = arr[b], arr[a]

    def block_swap_fw(self, arr, a, b, n):
        for i in range(n):
            self.swap(arr, a+i, b+i)

    def block_swap_bw(self, arr, a, b, n): # Only used in backwards galloping to prevent range overlap causing issues
        for i in range(n):
            self.swap(arr, a-1-i, b-1-i)

    def log_phi(self, n):
        log2 = 0
        while n:
            log2 += 1
            n >>= 1
        return (log2*369)>>8 # A pretty good approximation of log base phi

    def block_size(self, length, memory):
        disc = memory**2-4*length
        if memory > int(length**0.5) and memory < int(length**0.5)+length//int(length**0.5):
            return memory
        if disc < 0:
            return int(length**0.5)
        b = (memory+disc**0.5)/2
        return int(b+0.5)

    def initialise(self, arr, start, end):
        n = end-start
        if self.memory_size < int(n**0.5):
            self.strategy = 0 # Fully in-place
        elif self.memory_size < int(n**0.5)+(n//int(n**0.5)):
            self.strategy = 1 # External buffer
        elif self.memory_size < n//2:
            self.strategy = 2 # External buffer and tags
        else:
            self.memory_size = n//2 # Don't need any more, so clamp it here
            self.strategy = 3 # Full auxiliary

        self.block_len = self.block_size(n, self.memory_size)

        stack_len = self.log_phi(n >> 3) # MIN_RUN/2 should be greater than 16 otherwise there will be overflow
        self.run_start = Array(stack_len)
        self.run_end = Array(stack_len)
        self.run_count = 0
        first_run_end = self.count_run(arr, start, end)
        if first_run_end == end:
            return True
        if first_run_end < self.min_run:
            self.exp_insert_sort(arr, start, start+self.min_run)
            first_run_end = start+self.min_run
        self.push_run(start, first_run_end)

        if self.strategy in (0, 1):
            tag_target = n // self.block_len
            buffer_target = self.block_len * (not self.strategy)
            self.external_buffer = Array(self.memory_size)
            print("Memory used: "+str(self.memory_size))
            self.external_tags = None
            self.tags, self.buffer = self.extract_buffers(arr, start, end, tag_target, buffer_target)

            if self.buf_left:
                self.tags_start = start
                self.buffer_start = start+self.tags
            else:
                self.tags_start = end-self.buffer-self.tags
                self.buffer_start = end-self.buffer
                
            self.tags_sufficient = self.tags >= tag_target
            
            if self.buf_left:
                self.merge_start = self.buffer_start+self.buffer
                self.merge_end = end
            else:
                self.merge_start = start
                self.merge_end = self.tags_start
            
            self.curr_start = self.merge_start
            if self.buf_left:
                if self.run_start[0] < self.merge_start:
                    if self.run_end[0] <= self.merge_start:
                        self.run_count = 0
                    else:
                        self.run_start[0] = self.merge_start
                        self.curr_start = self.run_end[0]
            else:
                self.run_count = 0 # No assumptions can be reliably made when the buffer is not inside the first run
        elif self.strategy == 2:
            self.merge_start = start
            self.curr_start = self.run_end[0]
            self.merge_end = end
            self.tags = self.buffer = 0
            self.tags_start = self.buffer_start = None
            self.tags_sufficient = True

            self.external_buffer = Array(self.block_len)
            self.external_tags = Array(n//self.block_len)
            print("Memory used: "+str(len(self.external_buffer)+len(self.external_tags)))
        else:
            self.merge_start = self.curr_start = start
            self.curr_start = self.run_end[0]
            self.merge_end = end
            self.tags = self.buffer = 0
            self.tags_start = self.buffer_start = None
            self.tags_sufficient = True

            self.external_buffer = Array(self.memory_size)
            print("Memory used: "+str(self.memory_size))
            self.external_tags = None
        return False

    def min_run_len(self, n):
        r = 0
        while n >= self.MIN_RUN:
            r |= n & 1
            n >>= 1
        return n+r

    def sort(self, arr, start, end): # Main loop
        n = end-start

        if end-start < self.MIN_RUN:
            self.count_run(arr, start, end)
            self.exp_insert_sort(arr, start, end)
            return
        self.min_run = self.min_run_len(n >> 3)
        
        if self.initialise(arr, start, end):
            return

        curr_start = self.curr_start
        merge_end = self.merge_end
        self.min_run = self.min_run_len(merge_end-self.merge_start)

        while curr_start < merge_end:
            next_end = self.count_run(arr, curr_start, merge_end)
            if next_end-curr_start < self.min_run:
                next_end = min(curr_start+self.min_run, merge_end)
                self.exp_insert_sort(arr, curr_start, next_end)
            self.push_run(curr_start, next_end)
            self.merge_collapse(arr)
            curr_start = next_end
        self.merge_force(arr)

        if not self.strategy in (0,1):
            return

        self.redistribute_buffers(arr, start, end)

    def redistribute_buffers(self, arr, start, end):
        tags_start = self.tags_start
        tags = self.tags
        buffer_start = self.buffer_start
        buffer = self.buffer

        if self.buf_left:
            if buffer <= self.MIN_RUN:
                self.exp_insert_sort(arr, buffer_start, buffer_start+buffer)
            else:
                self.shellsort(arr, buffer_start, buffer_start+buffer)
            self.lazy_stable_merge(arr, buffer_start, buffer_start+buffer, end, False)

            if tags <= self.MIN_RUN:
                self.exp_insert_sort(arr, tags_start, tags_start+min(tags, self.last_used))
            else:
                self.shellsort(arr, tags_start, tags_start+min(tags, self.last_used))
            self.lazy_stable_merge(arr, tags_start, tags_start+tags, end, False)
        else:
            self.rotate(arr, tags_start, buffer_start, end)
            buffer_start = tags_start
            tags_start += buffer

            if buffer <= self.MIN_RUN:
                self.exp_insert_sort(arr, buffer_start, buffer_start+buffer)
            else:
                self.shellsort(arr, buffer_start, buffer_start+buffer)
            self.lazy_stable_merge(arr, start, buffer_start, buffer_start+buffer, True)

            if tags <= self.MIN_RUN:
                self.exp_insert_sort(arr, tags_start, tags_start+min(tags, self.last_used))
            else:
                self.shellsort(arr, tags_start, tags_start+min(tags, self.last_used))
            self.lazy_stable_merge(arr, start, tags_start, tags_start+tags, True)

    def exp_insert_sort(self, arr, start, end): # Insertion sort in what looks to be three lines feels deeply disturbing...
        for i in range(start+1, end):
            pos = self.exp_search_bw(arr, arr[i], start, i, True)
            self.insert(arr, i, pos)

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

    def merge_decide(self, arr, start, mid, end): # Just making sure there's capacity and that the merge is in fact necessary
        if not self.merge_small:
            start, end, skip = self.check_bounds(arr, start, mid, end)
            if skip:
                return end

        left_len, right_len = mid-start, end-mid

        if min(left_len, right_len) <= self.memory_size:
            self.merge_ext(arr, start, mid, end, False)
        elif min(left_len, right_len) <= self.buffer:
            self.merge_buf(arr, start, mid, end)
        else:
            self.lazy_stable_merge(arr, start, mid, end, False)

        return end

    def lazy_stable_merge(self, arr, start, mid, end, right):
        if mid-start <= end-mid:
            s = start
            l = mid

            while l-s > self.memory_size and l < end:
                if (arr[s] >= arr[l] if right else arr[s] > arr[l]):
                    p = self.exp_search_fw(arr, arr[s], l, end, right)
                    self.rotate(arr, s, l, p)
                    s += p-l
                    l = p
                else:
                    s += 1
            if l-s > 0 and l < end:
                self.merge_ext_fw(arr, s, l, end, right)
        else:
            s = end - 1
            l = mid - 1

            while s+1-l >= self.memory_size and l >= start:
                if (arr[l] >= arr[s]) if right else (arr[l] > arr[s]):
                    p = self.exp_search_bw(arr, arr[s], start, l, not right)
                    self.rotate(arr, p, l+1, s+1)
                    s -= l+1-p
                    l = p-1
                else:
                    s -= 1
            if s+1-l > 0 and l >= start:
                self.merge_ext_bw(arr, start, l+1, s+1, right)

    def merge_buf(self, arr, start, mid, end): # Ensures merge direction matches buffer's capacity
        if mid-start <= end-mid:
            self.merge_buf_fw(arr, start, mid, end)
        else:
            self.merge_buf_bw(arr, start, mid, end)

    def merge_buf_fw(self, arr, start, mid, end): # The galloping merges are actually pretty useful for something you'd expect to only be working on sqrt n values...
        min_gallop = self.min_gallop
        count_left = count_right = 0

        buffer_start = self.buffer_start
        left_len = mid-start
        self.block_swap_fw(arr, start, buffer_start, left_len) # Move half into buffer

        i, j, dest = buffer_start, mid, start
        while i < buffer_start+left_len and j < end:
            while i < buffer_start+left_len and j < end and max(count_left, count_right) < min_gallop: # Normal merge phase
                if arr[i] <= arr[j]:
                    self.swap(arr, i, dest) # Using swaps to avoid deleting the buffer
                    count_left += 1; count_right = 0
                    i += 1
                else:
                    self.swap(arr, j, dest)
                    count_right += 1; count_left = 0
                    j += 1
                dest += 1

            while i < buffer_start+left_len and j < end: # Galloping merge phase
                count_left = self.exp_search_fw(arr, arr[j], i, buffer_start+left_len, True)-i
                if count_left:
                    self.block_swap_fw(arr, i, dest, count_left)
                    i += count_left; dest += count_left
                    if i >= buffer_start+left_len:
                        break
                self.swap(arr, j, dest)
                j += 1; dest += 1
                if j >= end:
                    break
                count_right = self.exp_search_fw(arr, arr[i], j, end, False)-j
                if count_right:
                    self.block_swap_fw(arr, j, dest, count_right)
                    j += count_right; dest += count_right
                    if j >= end:
                        break
                self.swap(arr, i, dest)
                i += 1; dest += 1
                if i >= buffer_start+left_len:
                    break

                if max(count_left, count_right) < min_gallop:
                    min_gallop += 2
                    break

        while i < buffer_start+left_len:
            self.swap(arr, i, dest)
            i += 1
            dest += 1

        self.min_gallop = min_gallop

    def merge_buf_bw(self, arr, start, mid, end): # So you know what I said about the galloping merges? Not so much in this direction.
        min_gallop = self.min_gallop
        count_left = count_right = 0

        buffer_start = self.buffer_start
        right_len = end-mid
        self.block_swap_fw(arr, buffer_start, mid, right_len)

        i, j, dest = mid-1, buffer_start+right_len-1, end-1
        while i >= start and j >= buffer_start:
            while i >= start and j >= buffer_start and max(count_left, count_right) < min_gallop: # Normal merge phase
                if arr[j] >= arr[i]:
                    self.swap(arr, j, dest)
                    count_right += 1; count_left = 0
                    j -= 1
                else:
                    self.swap(arr, i, dest)
                    count_left += 1; count_right = 0
                    i -= 1
                dest -= 1

            while i >= start and j >= buffer_start: # Galloping phase
                count_left = i+1-self.exp_search_bw(arr, arr[j], start, i+1, True)
                if count_left:
                    self.block_swap_bw(arr, i+1, dest+1, count_left) # It took me so long to figure out that the direction of the block swap was actually important...
                    i -= count_left; dest -= count_left
                    if i < start:
                        break

                self.swap(arr, j, dest)
                j -= 1; dest -= 1
                if j < buffer_start:
                    break

                count_right = j+1-self.exp_search_bw(arr, arr[i], buffer_start, j+1, False)
                if count_right:
                    self.block_swap_bw(arr, j+1, dest+1, count_right)
                    j -= count_right; dest -= count_right
                    if j < buffer_start:
                        break

                self.swap(arr, i, dest)
                i -= 1; dest -= 1
                if i < start:
                    break

                if max(count_left, count_right) < min_gallop:
                    min_gallop += 2 # Penalise when galloping is exited
                    break

        while j >= buffer_start: # Copy any remaining elements in buffer
            self.swap(arr, j, dest)
            j -= 1
            dest -= 1

        self.min_gallop = min_gallop # Globalise min_gallop

    def arr_copy_fw(self, src, a, dest, b, n):
        for i in range(n):
            dest[b+i] = src[a+i]

    def arr_copy_bw(self, src, a, dest, b, n):
        a -= 1; b -= 1
        for i in range(n):
            dest[b-i] = src[a-i]

    def merge_ext(self, arr, start, mid, end, right):
        if mid-start <= end-mid:
            self.merge_ext_fw(arr, start, mid, end, right)
        else:
            self.merge_ext_bw(arr, start, mid, end, right)

    def merge_ext_fw(self, arr, start, mid, end, right):
        buffer = self.external_buffer
        min_gallop = self.min_gallop
        count_left = count_right = 0

        left_len = mid-start
        self.arr_copy_fw(arr, start, buffer, 0, left_len)

        i, j, dest = 0, mid, start
        while i < left_len and j < end:
            while i < left_len and j < end and max(count_left, count_right) < min_gallop:
                if (buffer[i] < arr[j] if right else buffer[i] <= arr[j]):
                    arr[dest] = buffer[i]
                    count_left += 1; count_right = 0
                    i += 1
                else:
                    arr[dest] = arr[j]
                    count_right += 1; count_left = 0
                    j += 1
                dest += 1

            while i < left_len and j < end:
                count_left = self.exp_search_fw(buffer, arr[j], i, left_len, not right)-i
                if count_left:
                    self.arr_copy_fw(buffer, i, arr, dest, count_left)
                    i += count_left; dest += count_left
                    if i >= left_len:
                        break
                arr[dest] = arr[j]
                j += 1; dest += 1
                if j >= end:
                    break
                count_right = self.exp_search_fw(arr, buffer[i], j, end, right)-j
                if count_right:
                    self.arr_copy_fw(arr, j, arr, dest, count_right)
                    j += count_right; dest += count_right
                    if j >= end:
                        break
                arr[dest] = buffer[i]
                i += 1; dest += 1
                if i >= left_len:
                    break

                if max(count_right, count_left) < min_gallop:
                    min_gallop += 2
                    break

        while i < left_len:
            arr[dest] = buffer[i]
            i += 1; dest += 1

        self.min_gallop = min_gallop

    def merge_ext_bw(self, arr, start, mid, end, right):
        buffer = self.external_buffer
        min_gallop = self.min_gallop
        count_left = count_right = 0

        right_len = end-mid
        self.arr_copy_fw(arr, mid, buffer, 0, right_len)

        i, j, dest = mid-1, right_len-1, end-1
        while i >= start and j >= 0:
            while i >= start and j >= 0 and max(count_left, count_right) < min_gallop:
                if (buffer[j] > arr[i] if right else buffer[j] >= arr[i]):
                    arr[dest] = buffer[j]
                    count_right += 1; count_left = 0
                    j -= 1
                else:
                    arr[dest] = arr[i]
                    count_left += 1; count_right = 0
                    i -= 1
                dest -= 1

            while i >= start and j >= 0:
                count_left = i+1-self.exp_search_bw(arr, buffer[j], start, i+1, not right)
                if count_left:
                    self.arr_copy_bw(arr, i+1, arr, dest+1, count_left)
                    i -= count_left; dest -= count_left
                    if i < start:
                        break

                arr[dest] = buffer[j]
                j -= 1; dest -= 1
                if j < 0:
                    break

                count_right = j+1-self.exp_search_bw(buffer, arr[i], 0, j+1, right)
                if count_right:
                    self.arr_copy_bw(buffer, j+1, arr, dest+1, count_right)
                    j -= count_right; dest -= count_right
                    if j < 0:
                        break

                arr[dest] = arr[i]
                i -= 1; dest -= 1
                if i < start:
                    break

                if max(count_left, count_right) < min_gallop:
                    min_gallop += 2
                    break

        while j >= 0:
            arr[dest] = buffer[j]
            j -= 1; dest -= 1

        self.min_gallop = min_gallop

    def reset_tags(self, arr, tags):
        tags_start = self.tags_start
        length = min(tags, self.last_used)
        if length <= self.MIN_RUN:
            self.exp_insert_sort(arr, tags_start, tags_start+length)
        else:
            self.shellsort(arr, tags_start, tags_start+length)
        self.last_used = max(tags, self.last_used)

    def block_merge(self, arr, start, mid, end):
        tags_start, tags = self.tags_start, self.tags
        buffer_start, buffer = self.buffer_start, self.buffer
        block_len = self.block_len

        len_a, len_b = mid-start, end-mid

        block_count_a = len_a//block_len
        block_count_b = len_b//block_len

        block_count = block_count_a+block_count_b

        remainder_a = len_a%block_len # Leftover elements in A half (at start)
        remainder_b = len_b%block_len # Leftover elements in B half (at end)

        blocks_start = start+remainder_a

        if self.strategy == 2:
            for i in range(block_count):
                self.external_tags[i] = i
            mid_tag = self.external_tags[block_count_a]
        else:
            self.reset_tags(arr, block_count)
            mid_tag = arr[tags_start+block_count_a] # Marker for first block in B half

        self.block_select_sort(arr, blocks_start, block_count, block_len, block_count_a, mid_tag) # Sort all full blocks between remainders
        self.cleanup_blocks(arr, start, end, block_count, block_len, remainder_a, remainder_b, mid_tag) 

    def get_tag(self, arr, tag):
        if self.strategy == 2:
            return self.external_tags[tag]
        else:
            return arr[self.tags_start+tag]

    def block_select_sort(self, arr, start, block_count, block_len, block_b, mid_tag):
        tags_start = self.tags_start
        b_end = block_b+1 # Only look at blocks if the min is guaranteed to be here

        for i in range(block_count):
            min_idx = i

            if self.get_tag(arr, i) < mid_tag: # Determining value used to compare blocks based on which half they came from to preserve stability
                min_val = arr[start+i*block_len] # First item of A block
            else:
                min_val = arr[start+(i+1)*block_len-1] # Last item of B block

            for j in range(max(block_b,i+1), b_end):
                if self.get_tag(arr, j) < mid_tag:
                    block_val = arr[start+j*block_len]
                else:
                    block_val = arr[start+(j+1)*block_len-1]

                if block_val < min_val or (block_val == min_val and self.get_tag(arr, j) < self.get_tag(arr, min_idx)):
                    min_val = block_val
                    min_idx = j

            if min_idx != i:
                self.block_swap_fw(arr, start+i*block_len, start+min_idx*block_len, block_len)
                if self.strategy == 2:
                    self.swap(self.external_tags, i, min_idx)
                else:
                    self.swap(arr, tags_start+i, tags_start+min_idx)

                if b_end < block_count: # Only increase range being checked when a swap has been made
                    b_end += 1

    def cleanup_blocks(self, arr, start, end, block_count, block_len, left_remainder, right_remainder, mid_tag):
        tags_start = self.tags_start
        buffer_start = self.buffer_start

        merge_end = end
        block_end = end

        if right_remainder > 0:
            right_start = block_end-right_remainder
            merge_end = self.merge_decide(arr, right_start-block_len, right_start, merge_end)
            block_end -= right_remainder
            last = block_count-1
        else:
            block_end -= block_len
            last = block_count-2

        for i in range(last, -1, -1): # Doing this backwards is the simplest way to do it stably (I was not about to spend four years learning fragment logic)
            block_start = block_end-block_len
            if self.get_tag(arr, i) < mid_tag:
                merge_end = self.merge_decide(arr, block_start, block_end, merge_end) # Shrink end to preserve stability (and let merge checks do less work)
            block_end -= block_len

        if left_remainder > 0:
            self.merge_decide(arr, start, start+left_remainder, merge_end)

    def buf_search(self, arr, target, start, end):
        if target > arr[end-1]:
            return end, False
        if target < arr[start]:
            return start, False
        end -= 1
        while start <= end:
            mid = (start+end)//2
            if arr[mid] == target:
                return mid, True
            elif arr[mid] > target:
                end = mid-1
            else:
                start = mid+1

        return start, False

    def extract_buffers(self, arr, start, end, tag_target, buffer_target):
        speed = UniV_getSpeed()
        
        tags_start = start
        buffer_start = start+1
        tag_count = 1
        buffer_count = 0

        i = start + 1
        while i < end:
            x = arr[i]
            if tag_count < tag_target:
                pos, exists = self.buf_search(arr, x, tags_start, tags_start+tag_count) # Check in tags with binary search
                if not exists:
                    self.insert(arr, i, pos)
                    tag_count += 1
                    buffer_start += 1
                    i += 1
                    continue

            if buffer_count < buffer_target:
                pos, exists = self.buf_search(arr, x, buffer_start, buffer_start+buffer_count) # Check in buffer with binary search
                if not exists:
                    self.insert(arr, i, pos)
                    buffer_count += 1
                    i += 1
                    continue

            if tag_count >= tag_target and buffer_count >= buffer_target: # If both full, exit early
                break

            i += 1 # Not in either, skip this item and switch to scanning
            
            UniV_setSpeed(speed*6) # This step just looks very slow
            while i < end: # Special case: Non-unique in both buffers, switch to scanning and rotating to keep O(n log n)
                prev = x
                x = arr[i]

                if prev == x:
                    i += 1
                    continue

                pos, exists = self.buf_search(arr, x, tags_start, tags_start+tag_count)
                if not exists and tag_count < tag_target:
                    self.rotate(arr, tags_start, buffer_start+buffer_count, i) # Only move once a unique item is found - reduces overall writes to O(n)
                    offset = i-(buffer_start+buffer_count)
                    pos += offset
                    tags_start += offset
                    buffer_start += offset

                    self.insert(arr, i, pos)
                    tag_count += 1
                    buffer_start += 1

                    dup_streak = 0
                    i += 1
                    break

                pos, exists = self.buf_search(arr, x, buffer_start, buffer_start+buffer_count)
                if not exists and buffer_count < buffer_target:
                    self.rotate(arr, tags_start, buffer_start+buffer_count, i)
                    offset = i-(buffer_start+buffer_count)
                    pos += offset
                    tags_start += offset
                    buffer_start += offset

                    self.insert(arr, i, pos)
                    buffer_count += 1

                    dup_streak = 0
                    i += 1
                    break

                i += 1
            
            UniV_setSpeed(speed)
        self.buf_left = True
        if tags_start > start:
            if buffer_start <= (start+end)//2 or tag_count < 4:
                self.rotate(arr, start, tags_start, buffer_start+buffer_count) # Move buffer to start if less than halfway
            else:
                self.rotate(arr, tags_start, buffer_start+buffer_count, end) # Move buffer to end otherwise
                self.buf_left = False

        return tag_count, buffer_count

    def insert(self, arr, src, dest):
        itm = arr[src]

        if src < dest:
            for i in range(src, dest):
                arr[i] = arr[i+1]
        elif src > dest:
            for i in range(src, dest, -1):
                arr[i] = arr[i-1]
        arr[dest] = itm

    def rotate(self, arr, a, m, e): # Trinity rotation
        len_a = m-a
        len_b = e-m

        if len_a < 1 or len_b < 1:
            return

        if len_a == len_b: # Gries-Mills' best-case
            self.block_swap_fw(arr, a, m, len_a)
            return

        if min(len_a, len_b) <= min(self.memory_size, 16): # Bridge rotation (faster than cycle reversal when largely unbalanced, but uses auxiliary space)
            tmp = self.external_buffer
            if len_a <= len_b:
                for i in range(len_a):
                    tmp[i] = arr[a+i]
                for i in range(len_b):
                    arr[a+i] = arr[m+i]
                for i in range(len_a):
                    arr[a+len_b+i] = tmp[i]
            else:
                for i in range(len_b):
                    tmp[i] = arr[m+i]
                for i in range(len_a):
                    arr[e-1-i] = arr[a+len_a-1-i]
                for i in range(len_b):
                    arr[a+i] = tmp[i]
            return

        b = m-1
        c = m
        d = e-1

        tmp = 0

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

    def reverse(self, arr, start, end):
        end -= 1 # Exclusive end
        while start < end:
            self.swap(arr, start, end)
            start += 1
            end -= 1

    def exp_search_fw(self, arr, target, start, end, right):
        if (arr[start] > target if right else arr[start] >= target):
            return start
        i = 1
        while start+i < end and (arr[start+i] <= target if right else arr[start+i] < target):
            i *= 2
        low = start+(i//2)
        high = min(start+i, end)
        while low < high:
            mid = (low+high)//2
            if (arr[mid] <= target if right else arr[mid] < target):
                low = mid+1
            else:
                high = mid
        return low

    def exp_search_bw(self, arr, target, start, end, right):
        if (arr[end-1] <= target if right else arr[end-1] < target):
            return end
        i = 1
        while end-1-i >= start and (arr[end-1-i] > target if right else arr[end-1-i] >= target):
            i *= 2
        low = max(end-1-i, start)
        high = end-1-(i//2)
        while low < high:
            mid = (low+high)//2
            if (arr[mid] > target if right else arr[mid] >= target):
                high = mid
            else:
                low = mid+1
        return low

    def check_bounds(self, arr, start, mid, end):
        if start >= mid or mid >= end or start >= end:
            return start, end, True

        if arr[mid-1] <= arr[mid]:
            return start, end, True

        if mid-start >= 8 or end-mid >= 8:
            start = self.exp_search_fw(arr, arr[mid], start, mid, True)
            end = self.exp_search_bw(arr, arr[mid-1], mid, end, False)

        if arr[start] > arr[end-1]:
            self.rotate(arr, start, mid, end)
            return start, end, True

        return start, end, False
    
    def count_run(self, arr, start, end):
        n = end-start
        if n <= 0:
            return start
        
        i = 1
        while i < n and arr[start + i] >= arr[start+i-1]:
            i += 1

        if i >= n or (i > 1 and arr[start] < arr[start+i-1]):
            return start+i

        head = i
        l = i

        while True:
            i += 1
            while i < n and arr[start+i] >= arr[start+i-1]:
                i += 1
            
            if i > l+1 and arr[start+l] < arr[start+i-1]:
                p = self.exp_search_fw(arr, arr[start+l], start+l, start+i, True)-start
                self.reverse(arr, start+l, start+p)
                l = p
                break
            
            self.reverse(arr, start+l, start+i)
            l = i
            
            if i >= n:
                break

        if head < l:
            self.reverse(arr, start, start+head)
            self.reverse(arr, start, start+l)
        
        return start+l

    def run_len(self, i):
        return self.run_end[i]-self.run_start[i]

    def merge_collapse(self, arr): # Borrowed from Timsort
        while self.run_count > 1:
            n = self.run_count-2

            if n >= 1 and self.run_len(n-1) <= self.run_len(n)+self.run_len(n+1) or n >= 2 and self.run_len(n-2) <= self.run_len(n)+self.run_len(n-1):
                if self.run_len(n-1) < self.run_len(n+1):
                    n -= 1
            elif self.run_len(n) > self.run_len(n+1):
                break
            self.merge_at(arr, n)

    def merge_force(self, arr): # Also borrowed from Timsort
        while self.run_count > 1:
            n = self.run_count-2

            if n > 0 and self.run_len(n-1) < self.run_len(n+1):
                n -= 1

            self.merge_at(arr, n)

    def merge_at(self, arr, n): # Guess what? It's also borrowed from Timsort
        start_a, end_a = self.run_start[n], self.run_end[n]
        start_b, end_b = self.run_start[n+1], self.run_end[n+1]

        self.run_end[n] = end_b

        if n == self.run_count-3:
            self.run_start[n+1] = self.run_start[n+2]
            self.run_end[n+1] = self.run_end[n+2]

        self.run_count -= 1

        start_a, end_b, skip = self.check_bounds(arr, start_a, start_b, end_b)

        if skip:
            return

        if self.strategy in (0, 1) and self.tags < 4:
            self.lazy_stable_merge(arr, start_a, start_b, end_b, False)
            return
        
        buffer = self.buffer if self.strategy == 0 else len(self.external_buffer)
        if not self.tags_sufficient and max(end_a-start_a, end_b-start_b) > buffer:
            if (end_b-start_a)//buffer > self.tags:
                self.block_len = (end_b-start_a)//self.tags
            else:
                self.block_len = buffer

        left_len, right_len = end_a-start_a, end_b-start_b

        if self.strategy == 3:
            self.merge_ext(arr, start_a, start_b, end_b, False)
        elif min(left_len, right_len) > buffer:
            self.merge_small = False
            self.block_merge(arr, start_a, start_b, end_b)
        else:
            self.merge_small = True
            self.merge_decide(arr, start_a, start_b, end_b)

    def push_run(self, start, end):
        self.run_start[self.run_count], self.run_end[self.run_count] = start, end
        self.run_count += 1

@Sort("Block Merge Sorts", "miausort", "miausort") # Just for UniV (visualiser)
def miausort(arr):
    n = len(arr)
    mem = UniV_getUserInput("Enter memory (0 to -3 for different strategies)", "0", parseInt)
    if mem == 0 or mem < -3:
        mem = 16
    elif mem == -1:
        mem = n**0.5
    elif mem == -2:
        mem = n**0.5+n//(n**0.5)
    elif mem == -3:
        mem = n//2
    MiauBlockMerge(mem).sort(arr, 0, len(arr))