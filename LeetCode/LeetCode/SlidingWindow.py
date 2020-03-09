from typing import List
import collections
import math


#######################################################################################################################
# sliding window

class Solution_3:
    def lengthOfLongestSubstring(self, s: str) -> int:
        lo = hi = 0
        cnt = collections.defaultdict(int)
        res = 0
        while hi < len(s):
            cnt[s[hi]] += 1
            while cnt[s[hi]] > 1:
                cnt[s[lo]] -= 1
                lo += 1
            res = max(res, hi - lo + 1)
            hi += 1
        return res

class Solution_76:
    def minWindow(self, s: str, t: str) -> str:

        def counter(cq, ct):
            for k, v in ct.items():
                if k not in cq or cq[k] < v:
                    return False
            return True

        counter_t = collections.Counter(t)
        counter_q = {}

        if len(s) < len(t):
            return ""

        que = collections.deque()
        mini = float('inf')
        idx = 0
        for i, c in enumerate(s):
            que.append(c)
            counter_q[c] = counter_q.get(c, 0) + 1
            while que and counter(counter_q, counter_t):
                if len(que) < mini:
                    mini = len(que)
                    idx = i

                tmp = que.popleft()
                counter_q[tmp] -= 1

        if mini < float('inf'):
            return s[idx - mini + 1:idx + 1]
        else:
            return ''

    def minWindow(self, s: str, t: str) -> str:
        def check(cnt):
            return all(v <= 0 for v in cnt.values())

        lo = hi = 0
        counter = collections.Counter(t)
        min_size = float('inf')
        idx = -1
        while hi < len(s):
            counter[s[hi]] -= 1
            while lo <= hi and check(counter):
                if hi - lo + 1 < min_size:
                    min_size = hi - lo + 1
                    idx = lo
                counter[s[lo]] += 1
                lo += 1
            hi += 1

        if min_size < float('inf'):
            return s[idx:idx + min_size]
        else:
            return ''

class Solution_209:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        lo = hi = 0
        cur = 0
        res = float('inf')
        while hi < len(nums):
            cur += nums[hi]
            while cur >= s:
                res = min(res, hi - lo + 1)
                cur -= nums[lo]
                lo += 1

            hi += 1

        return res if res < float('inf') else 0

    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        # binary search
        sums = [0]
        for x in nums:
            sums.append(x + sums[-1])
        sums = sums[1:]
        res = float('inf')
        for i in range(len(sums)):
            lo = i
            hi = len(nums) - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                t = sums[mid] - (sums[i - 1] if i > 0 else 0)
                if t >= s:
                    hi = mid
                else:
                    lo = mid + 1
            # print(lo, hi)
            if sums[hi] - (sums[i - 1] if i > 0 else 0) >= s:
                res = min(res, lo + 1 - i)
        return res if res < float('inf') else 0


class Solution_424:
    # 滑动窗口
    def characterReplacement(self, s: str, k: int) -> int:
        lo = hi = 0
        count = collections.defaultdict(int)
        res = 0
        max_count = 0
        while hi < len(s):
            count[s[hi]] += 1
            # max_count 表示窗口内的字符数的最大值，
            # 注意有些时候 max_count 可能是 invalid (比如我们右移了窗口左边使得最大字符数减少了的时候)
            # 但没关系，这之前一定已经遇到了一个更长的子串。
            max_count = max(max_count, count[s[hi]])

            while (hi - lo + 1) - max_count > k:  # 当前窗口内其他字符个数大于 k 个
                count[s[lo]] -= 1
                lo += 1

            res = max(res, hi - lo + 1)
            hi += 1
        return res


class Solution_713:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # 需要严格 < k
        # opt(j) 表示 满足 nums[i] * ... * nums[j] 的最小 i 的值,
        # 我们需要找到一个能够 O(1) 时间获得 i~j 乘积的办法，首先想到前缀积，
        # 然后一个问题是，这里的数值非常到，所以我们可以考虑使用 log， 将前缀和
        # 变成前缀积

        # 注意，log 之后是浮点数，不能精确比较，使用 eps = 1e-9
        if k <= 0:
            return 0
        k = math.log(k)
        prefix = [0]
        for i in range(len(nums)):
            prefix.append(prefix[-1] + math.log(nums[i]))

        res = 0
        for j in range(len(nums)):
            # binary search
            lo, hi = 0, j + 1
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if prefix[j + 1] - prefix[mid] < k - 1e-9:
                    hi = mid
                else:
                    lo = mid + 1
            # print(lo, j)
            res += (j - lo + 1)

        return res

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1: return 0
        prod = 1
        res = lo = 0
        for hi, x in enumerate(nums):
            prod *= x
            while prod >= k:
                prod /= nums[lo]
                lo += 1
            res += (hi - lo + 1)
        return res

class Solution_904:
    def totalFruit(self, tree: List[int]) -> int:
        # 第 i 棵树有果实类型 tree[i]
        # 从任意一个数字出发，我们最多能收集两种果实，能收集的最大数量是多少

        # 先试试暴力方法, 果不其然TLE
        res = 0
        for i in range(len(tree)):
            buskets = {}
            for j in range(i, len(tree)):
                if tree[j] in buskets:
                    buskets[tree[j]] += 1
                else:
                    if len(buskets) >= 2:
                        break
                    else:
                        buskets[tree[j]] = 1
            # print(buskets)
            res = max(res, sum(buskets.values()))

        return res

    def totalFruit(self, tree: List[int]) -> int:
        res = lo = hi = 0
        types = collections.defaultdict(int)
        while hi < len(tree):
            types[tree[hi]] += 1
            while len(types) > 2:
                types[tree[lo]] -= 1
                if types[tree[lo]] == 0:
                    types.pop(tree[lo])
                lo += 1
            res = max(res, hi - lo + 1)
            hi += 1
        return res


class Solution_930:
    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        # 前缀和
        prefix = [0]
        for x in A:
            prefix.append(prefix[-1] + x)

        count = collections.Counter()
        # 找出所有 i<j 满足 prefix[j] - prefix[i] == S
        # 数 prefix[j] = prefix[i] + S, 所以就变成了我们遇到过多少 prefix[i] + S

        ans = 0
        for x in prefix:
            ans += count[x]
            count[x + S] += 1
        return ans

    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        # 一边计算前缀和一边计算结果数
        count = collections.Counter({0: 1})
        prefix = res = 0
        for x in A:
            prefix += x
            res += count[prefix - S]
            count[prefix] += 1
        return res

    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        def atMost(K):
            if K < 0:
                return 0
            res = lo = hi = 0
            while hi < len(A):
                K -= A[hi]
                while K < 0:  # 这里不用 bound lo
                    K += A[lo]
                    lo += 1
                res += hi - lo + 1
                hi += 1
            return res

        # 恰好是 S = 最多 S - 最多 (S-1)
        return atMost(S) - atMost(S - 1)


class Solution_992:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        # 一个连续子数组的不同数字的数目恰好是 K
        # 我们用滑动窗口求得不同数组数目最多为 K 和最多为 K-1 的情况，
        # 那么数目恰好为 K 等于 atMost(K) - atMost(k-1)

        def atMost(K):
            res = lo = hi = 0
            cnt = collections.defaultdict(int)
            while hi < len(A):
                cnt[A[hi]] += 1
                while len(cnt) > K:
                    cnt[A[lo]] -= 1
                    if cnt[A[lo]] == 0:
                        cnt.pop(A[lo])
                    lo += 1
                res += hi - lo + 1
                hi += 1
            return res

        return atMost(K) - atMost(K - 1)


import bisect


class Solution_1004:
    def longestOnes(self, A: List[int], K: int) -> int:
        # 找到最多包含 K 个零的最长连续子数组
        lo = hi = 0
        res = 0
        count = 0
        while hi < len(A):
            count += 1 if A[hi] == 0 else 0
            while lo <= hi and count > K:
                count -= 1 if A[lo] == 0 else 0
                lo += 1
            res = max(res, hi - lo + 1)
            hi += 1
        return res

    def longestOnes(self, A: List[int], K: int) -> int:
        lo = 0
        for hi in range(len(A)):
            K -= 1 - A[hi]
            if K < 0:
                K += 1 - A[lo]
                lo += 1
        return hi - lo + 1


class Solution_1234:
    def balancedString(self, s: str) -> int:
        # 使用 双指针 来保证窗口之外的其他字符 <= n/4

        lo = hi = 0
        res = len(s)
        count = collections.Counter(s)
        while hi < len(s):
            count[s[hi]] -= 1
            # 为什么这里 bound lo < len(s) 而不是 lo <= hi
            # 使用 lo <= hi 的时候要单独处理本来就是平衡的 case
            # 而是用 lo < len(s) 时当 lo > hi，原本平衡的 case 会导致 all() 为 false
            while lo < len(s) and all(len(s) / 4 >= count[c] for c in "QWER"):
                res = min(res, hi - lo + 1)
                count[s[lo]] += 1
                lo += 1
            hi += 1
        return res


class Solution_1248:
    def numberOfSubarrays(self, A, k):
        # 使用前缀和来算，偶数为 0，奇数为 1
        n = len(A)
        s = [0] * (n + 1)
        for i in range(1, n + 1):
            s[i] = s[i - 1] + A[i - 1] % 2

        # 前缀和是有序数组，我们可以使用二分查找
        ret = 0
        for i in range(n):
            ret += bisect.bisect_right(s, s[i] + k) - bisect.bisect_left(s, s[i] + k)
        return ret

    def numberOfSubarrays(self, A, k):
        # 滑动窗口，
        lo = hi = 0
        res = count = 0
        odds = 0
        while hi < len(A):
            if A[hi] & 1:
                odds += 1
                count = 0
            # count 记录了恰好包含 k 个奇数的（左边）子数组个数
            # 当我们继续向右扩展，遇到的是个偶数时，此时对应也有 count 个子数组
            while lo <= hi and odds == k:
                count += 1
                odds -= A[lo] & 1
                lo += 1
            res += count
            hi += 1
        return res

    def numberOfSubarrays(self, A, k):
        def atMost(k):
            res = lo = hi = 0
            while hi < len(A):
                k -= A[hi] & 1
                while k < 0:  # 这里依然不需要 bound lo，因为 lo >= hi 时, k = 0
                    k += A[lo] & 1
                    lo += 1
                # 求最多为 K 的情况,
                res += hi - lo + 1
                hi += 1
            return res

        return atMost(k) - atMost(k - 1)


class Solution_438:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 滑动窗口，就是找到窗口内的字符数恰好等于 p 的字符
        p_cnt = collections.Counter(p)
        s_cnt = collections.Counter()

        def check(s_cnt, p_cnt):
            for k, v in p_cnt.items():
                if k not in s_cnt:
                    return False
                if s_cnt[k] < v:
                    return False
            return True

        lo = hi = 0
        res = []
        while hi < len(s):
            s_cnt[s[hi]] += 1

            while lo <= hi and check(s_cnt, p_cnt):
                if len(s_cnt) == len(p_cnt) and all(s_cnt[k] == v for k, v in p_cnt.items()):
                    res.append(lo)
                s_cnt[s[lo]] -= 1
                if s_cnt[s[lo]] == 0:
                    s_cnt.pop(s[lo])
                lo += 1

            hi += 1
        return res

    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 我是想不出这种解法的
        # https://leetcode.com/problems/find-all-anagrams-in-a-string/discuss/92017/Python-O(n)-sliding-window-with-a-lot-of-comments.-Accepted-solution
        p_cnt = collections.Counter(p)
        lo = hi = cnt = 0
        res = []
        while hi < len(s):
            p_cnt[s[hi]] -= 1
            if p_cnt[s[hi]] >= 0:  # 如果是负的话，说明至今删去窗口内含有的字符不足以满足答案，
                cnt += 1           # 否则我们就可更新我们统计至当前为止 cross off 的字符数

            if hi > len(p) - 1:  # hi >= len(p)
                p_cnt[s[lo]] += 1
                if p_cnt[s[lo]] > 0:
                    cnt -= 1
                lo += 1

            if cnt == len(p):  # 如果恰好等于 p 的长度，那么当前窗口是一个答案
                res.append(lo)
            hi += 1
        return res


class Solution_567:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 和 438 一样的题目
        lo = hi = 0
        a = [0] * 26
        b = [0] * 26
        for c in s1:
            a[ord(c) - ord('a')] += 1

        while hi < len(s2):
            b[ord(s2[hi]) - ord('a')] += 1
            while lo <= hi and all(bx >= ax for ax, bx in zip(a, b)):
                if all(bx == ax for ax, bx in zip(a, b)):
                    return True
                b[ord(s2[lo]) - ord('a')] -= 1
                lo += 1

            hi += 1
        return False

    def checkInclusion(self, s1: str, s2: str) -> bool:
        # pass
        lo = hi = c = 0
        cnt = collections.Counter(s1)
        while hi < len(s2):
            cnt[s2[hi]] -= 1
            if cnt[s2[hi]] >= 0:
                c += 1
            if hi >= len(s1):
                cnt[s2[lo]] += 1
                if cnt[s2[lo]] > 0:
                    c -= 1
                lo += 1
            if c == len(s1):
                return True

            hi += 1
        return False