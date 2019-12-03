from typing import List
import collections
import itertools
import functools
import math
import heapq


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __repr__(self):
        return f'{self.val}, {{next:{self.next}}}'


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __repr__(self):
        return f'{{{self.val}, left:{{{self.left}}}, right{{{self.right}}}}}'


def coin():
    MOD = int(1e9 + 7)

    def f(n):
        v = [1, 2, 5, 10]
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        for x in v:
            for i in range(x, n + 1):
                if i >= x:
                    dp[i] = (dp[i] + dp[i - x]) % MOD
        return dp[n]

    n = int(input())
    print(f(n))


def max_product(nums):
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    left = [0] * len(nums)
    dp[0] = nums[0]
    left[0] = min(nums[0], 0)
    m = max(dp[0], 0)
    for i, x in enumerate(nums[1:]):
        index = i + 1
        dp[i + 1] = max(x * dp[i], x, x * left[i])
        if m < dp[i + 1]:
            m = dp[i + 1]
        left[i + 1] = min(x * dp[i], x, x * left[i], 0)
    print(dp)
    print(left)
    print(m)


# max_product([-2, 0, -1])


def unknownStartDay(day, month, year):
    def hasLeapDay(year):
        return 1 if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else 0

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # days since 31, 12, 1970
    def days_since_start(day, month, year):
        numDays = 0
        for y in range(year - 1, 1970, -1):
            numDays += 365 + hasLeapDay(y)
        numDays += sum(days_in_month[:month - 1])
        numDays += day
        if month > 2:
            numDays += hasLeapDay(year)
        return numDays

    knownStart = days_since_start(30, 9, 2019)
    d = days_since_start(day, month, year)
    print(d)
    print(knownStart)
    return day_names[(d - knownStart) % 7]


# print(unknownStartDay(31, 8, 2019))
# d = {}
# rd = {}
# cum = 1
# for i in range(2, 11):
#     cum *= i
#     d[cum] = i
#     rd[i] = cum
#
# n = int(input())
#
# count = []
# if n == 1:
#     s = input()
#     print(s[::-1])
# else:
#     x = d[n+1]
#     for i in range(x):
#         count.append({})
#
#     for i in range(n):
#         s = input()
#         for j, c in enumerate(s):
#             if c in count[j]:
#                 count[j][c] += 1
#             else:
#                 count[j][c] = 1
#     res = []
#     for i in range(d[n+1]):
#         for k, v in count[i].items():
#             if 0 < v < rd[x - 1]:
#                 res.append(k)
#     print(''.join(res))

####################################################################################
# fs = """5 10
# 0 2 9
# 3 0 9
# 4 5 7
# 0 0 9
# 0 0 2
# 3 2 10
# 0 5 2
# 4 8 1
# 6 0 9
# 0 7 1
# 10 0 8
# 0 0 3
# 9 0 6
# 0 0 9
# 0 0 3
# """
# fsr = fs.split('\n')
#
# tree1 = [[0, 0, 0] for _ in range(202)]
# tree2 = [[0, 0, 0] for _ in range(202)]
# raw = fsr[0]
# n, m = raw.split(' ')
# n, m = int(n), int(m)
# for i in range(n):
#     raw = fsr[i+1]
#     l, r, v = raw.split()
#     l, r, v = int(l), int(r), int(v)
#     tree1[i+1] = [l, v, r]
#
# for i in range(m):
#     raw = fsr[n+i+1]
#     l, r, v = raw.split()
#     l, r, v = int(l), int(r), int(v)
#     tree2[i+1] = [l, v, r]
#
# t = n
# def get_t():
#     global t
#     t = t+1
#     return t
#
# def pre(tree1, tree2, root1, root2):
#
#     if root1 > 0 and root2 > 0:
#         tree1[root1][1] += tree2[root2][1]
#         tree1[root1][0] = pre(tree1, tree2, tree1[root1][0], tree2[root2][0])
#         tree1[root1][2] = pre(tree1, tree2, tree1[root1][2], tree2[root2][2])
#         return root1
#     elif root1 > 0 and root2 <= 0:
#         return root1
#     elif root1 <= 0 and root2 > 0:
#         t = get_t()
#         print('###', root1, t, root2)
#         tree1[t][1] = tree2[root2][1]
#         tree1[t][0] = pre(tree1, tree2, tree1[t][0], tree2[root2][0])
#         tree1[t][2] = pre(tree1, tree2, tree1[t][2], tree2[root2][2])
#         return t
#     else:
#         return 0
#
#
# def level(tree, root):
#     que = [root]
#     res = []
#     while que:
#         print(que)
#         t = que[0]
#         que.pop(0)
#         res.append(tree[t][1])
#         if tree[t][0] > 0:
#             que.append(tree[t][0])
#         if tree[t][2] > 0:
#             que.append(tree[t][2])
#     return res
# print(level(tree1, 1))
# print(level(tree2, 1))
#
# pre(tree1, tree2, 1, 1)
# res = level(tree1, 1)
# for i, v in enumerate(res):
#     if i == len(res) - 1:
#         print(v)
#     else:
#         print(v, end=' ')

# n = int(input())
# t = [int(x) for x in input().split(' ')]
def increasingTriplet(self, nums) -> bool:
    if len(nums) < 3:
        return False
    i, j = 0, len(nums) - 1
    left, right = nums[0], nums[-1]
    left_cnt, r_cnt = 0, 0
    while i < len(nums) and j >= 0:
        if nums[i] > left:
            left_cnt += 1
            left = nums[i]
        if nums[j] < right:
            r_cnt += 1
            right = nums[j]
        i += 1
        j -= 1

    if left_cnt >= 2 or r_cnt >= 2:
        return True
    return False


"""
第一行一个正整数n(1<=n<=100)，表示二叉树有n个结点。
接下来n行，第i行两个整数li,ri (0<=li,ri<=n) ，分别表示第i个结点的左儿子和右儿子，为0代表空。

保证根为1，保证输入为合法二叉树。

第一行为二叉树的前序遍历；
第二行为中序遍历
第三行为后序遍历
第四行为层次遍历

每一行输出n个数，代表该方式遍历的结点的顺序，相邻两个数之间用一个空格相隔。
输入例子1:
5
3 2
0 5
0 4
0 0
0 0

输出例子1:
1 3 4 2 5
3 4 1 2 5
4 3 5 2 1
1 3 2 4 5
"""


### 7
def get_tree():
    tree = [{'l': 0, 'v': _, 'r': 0} for _ in range(101)]
    n = int(input())
    for i in range(n):
        l, r = input().split(' ')
        l, r = int(l), int(r)
        tree[i + 1]['l'] = l
        tree[i + 1]['r'] = r
    return tree, n


def pre_nonrec(tree, root, n):
    stack = []
    p = root
    res = []
    print(tree)
    i = 0
    while stack or p > 0:
        i += 1
        if i > n:
            break
        while p > 0:
            res.append(tree[p]['v'])
            stack.append(p)
            p = tree[p]['l']
        print(stack, p)

        if stack:
            p = stack[-1]
            stack.pop()
            if tree[p]['r'] > 0:
                p = tree[p]['r']

    return res


# print(pre_nonrec(tree, 1, n))


def findMin(nums) -> int:
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        print(left, mid, right)
        if nums[left] == nums[mid] == nums[right]:
            right -= 1
        elif nums[left] <= nums[mid] <= nums[right]:
            right = mid - 1
        elif nums[mid] <= nums[right] <= nums[left]:
            right = mid
        elif nums[mid] >= nums[left] >= nums[right]:
            left = mid
            if right - left == 1:
                left = right
    return nums[left]


# print(findMin([1, 10, 10, 10, 10]))
# r = [[3, 4], [1, 2], [2, 3]]
# r = sorted(r, key=lambda x: x[0])
# print(r)
# print(set('012340'))


def three():
    t = int(input())
    res = []
    for _ in range(t):
        n, m = input().split()
        n, m = int(n), int(m)
        if n >= 2 and m >= 2:
            res.append((n - 2) * (m - 2))
        elif n >= 2 and m < 2:
            res.append(n - 2)
        elif n < 2 and m >= 2:
            res.append(m - 2)
        else:
            res.append(1)
    for x in res:
        print(x)


def four():
    t = int(input())
    for _ in range(t):
        n, k = [int(x) for x in input().split(' ')]
        if n < 3:
            print(0, 0)
        else:
            if n >= 2 * k:
                print(0, k - 1)
            else:
                print(0, n - k)


# n, m = [int(t) for t in input().split(' ')]
# voter = [0]
# d = {}
# nsupporter = set()
# for i in range(n):
#     x, y = [int(t) for t in input().split(' ')]
#     voter.append((x, y))
#     if x != 1:
#         nsupporter.add(i + 1)
#     if x in d:
#         d[x].add(i + 1)
#     else:
#         d[x] = {i + 1}
#
#
# def get_winner():
#     winner = max(d, key=lambda k: len(d[k]))
#     return winner
#
#
# winner = get_winner()
# candy = 0
# while winner != 1:
#     min_s = min(d[winner], key=lambda x: voter[x][1])
#     min_ns = min(nsupporter, key=lambda x: voter[x][1])
#     if voter[min_s][1] > 2 * voter[min_ns][1]:
#         candy += voter[min_s][1]
#         d[winner] = d[winner] - {min_s}
#         d[1].add(min_s)
#     else:
#         candy += voter[min_ns][1]
#         d[voter[min_ns][0]] = d[voter[min_ns][0]] - {min_ns}
#         if 1 in d:
#             d[1].add(min_ns)
#         else:
#             d[1] = {min_ns}
#     winner = get_winner()
# print(candy)


class Solution:
    def lexicalOrder(self, n: int):
        def foo(tmp, i, n):
            if i <= n:
                tmp.append(i)
                for j in range(10):
                    foo(tmp, i * 10 + j, n)

        res = []
        for i in range(1, 10):
            tmp = []
            foo(tmp, i, n)
            res += tmp
        return res


def subsetsWithDup(nums):
    res = set()
    n = len(nums)

    def foo(i, tmp, nums):
        nonlocal res, n
        if i == n:
            res.add(tuple(tmp))
        else:
            foo(i + 1, tmp, nums)
            foo(i + 1, tmp + [nums[i]], nums)

    foo(0, [], nums)
    return list(res)


# print(subsetsWithDup([1,2,2]))


def numRollsToTarget(d: int, f: int, target: int) -> int:
    res = [0]
    MOD = int(1e9 + 7)

    for i in range(d):
        tmp = []
        for n in range(1, f + 1):
            for k in res:
                if n + k <= target:
                    tmp.append((n + k))

        res.extend(tmp)
    return res.count(target) % MOD


def _insert(ls, val):
    left, right = 0, len(ls) - 1
    if len(ls) == 1 and ls[0] > val:
        ls.append(val)
        return
    while left < right:
        mid = left + (right - left) // 2
        if val < ls[mid]:
            left = mid + 1
        else:
            right = mid
    print(left)

    ls.insert(left, val)


#
# L = [4 ,3 ,2, -float('inf')]
# _insert(L, 1)
# print(L)

class Solution:
    def findSolution(self, customfunction: 'CustomFunction', z: int):
        res = []

        def f(x, y):
            return x + y

        for x in range(1, 1001):
            lo, hi = 1, 1000
            while lo <= hi:
                y = (lo + hi) >> 1
                t = f(x, y)
                if t == z:
                    res.append([x, y])
                    break
                elif t < z:
                    lo = y + 1
                else:
                    hi = y - 1
        return res


class Solution_:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        res = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i - 1][j - 1] == '1':
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    res = max(res, dp[i][j])
        return res * res

    def maximalSqure(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return 0
        res = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    p, q = i, j
                    cur = 1

                    pass


#######################################################################################################################
# sliding window

class Solution_209:
    def _minSubArrayLen(self, s: int, nums: List[int]) -> int:
        # two pointer[sliding window]
        m = float('inf')
        left = 0
        t = 0
        for i, x in enumerate(nums):
            t += nums[i]
            while t >= s:
                m = min(m, i + 1 - left)
                t -= nums[left]
                left += 1
        return m if m < float('inf') else 0

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


class Solution_128:
    def _longestConsecutive(self, nums: List[int]) -> int:
        # 简单方法是排序后查找，这样复杂度是O(nlogn), AC
        if len(nums) == 0:
            return 0
        s = sorted(nums)
        head = cur = s[0]
        res = 0
        for i in range(1, len(s)):
            if s[i] > cur + 1:
                res = max(res, cur - head + 1)
                head = s[i]
            cur = s[i]
        res = max(res, cur - head + 1)
        return res

    def longestConsecutive(self, nums: List[int]) -> int:
        # 题目要求使用O(n)的方法，使用hashset+类似并查集的思想？
        mp = {}
        s = set(nums)

        def find(n):
            if n in mp and mp[n] != n + 1:
                return mp[n]
            if n + 1 in s:
                mp[n] = find(n + 1)
                return mp[n]
            return n + 1

        res = 0
        for x in nums:
            res = max(res, find(x) - x)
        return res

    def longestConsecutive(self, nums: List[int]) -> int:
        # 或者是使用set直接加速lookup
        s = set(nums)
        m = 0
        for x in s:
            if x - 1 not in s:  # 保证不找重复的序列
                cur = x
                cnt = 1
                while cur + 1 in s:
                    cur += 1
                    cnt += 1

                m = max(m, cnt)
        return m


class Solution_55:
    def _canJump(self, nums: List[int]) -> bool:
        t = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= t:
                t = i
        return t == 0

    def canJump(self, nums):
        pos = 0
        i = 0
        while i <= pos:
            pos = max(pos, i + nums[i])
            if pos >= len(nums) - 1:
                return True
            i += 1
        return False


class Solution_45:
    def jump(self, nums: List[int]) -> int:

        dp = [float('inf')] * len(nums)
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(1, nums[i] + 1):
                if i + j < len(nums):
                    dp[i + j] = min(dp[i + j], dp[i] + 1)
        # print(LeetCode)
        # DP方法TLE
        return int(dp[-1])

    def jump(self, nums: List[int]) -> int:
        # jump game I中的greedy思路，将其看作一个BFS问题，类似于到最后一个节点的最短路
        # 如[2, 3,1,1,4]可以转换为
        #       2           lv0
        #       3   1       lv1
        #       1   4       lv2
        if len(nums) < 2:  # 边界要单独处理
            return 0
        pos = 0
        lv = 0
        i = 0
        while i <= pos:
            lv += 1
            next_pos = pos
            while i <= pos:  # dfs访问当前层, 相当于这一层的全部入队列
                next_pos = max(next_pos, nums[i] + i)
                if next_pos >= len(nums) - 1:
                    return lv
                i += 1
            pos = next_pos

        return 0


class Solution_48:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        # non in-place
        newm = [[0] * n for _ in range(n)]
        for j in range(n):
            for i in range(n - 1, -1, -1):
                newm[j][n - 1 - i] = matrix[i][j]
        matrix[:] = newm

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # in-place 要分两步来做，首先transpose,然后再flip(时刻牢记转置是个非常有用的操作)
        # 例如：
        #    1 2 3
        #    4 5 6
        #    7 8 9
        # 转置为：
        #    1 4 7
        #    2 5 8
        #    3 6 9
        # 水平翻转行
        #    7 4 1
        #    8 5 2
        #    9 6 3
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(n):
            lo, hi = 0, n - 1
            while lo < hi:
                matrix[i][lo], matrix[i][hi] = matrix[i][hi], matrix[i][lo]
                lo += 1
                hi -= 1

    def rotate(self, matrix: List[List[int]]) -> None:
        # 还有ring by ring的方法，不过ring by ring的难点在于定位，没有转置再反转的方法直观。
        n = len(matrix)
        lo, hi = 0, n - 1
        while lo < hi:
            for i in range(hi - lo):
                matrix[lo][lo + i], matrix[lo + i][hi] = matrix[lo + i][hi], matrix[lo][lo + i]
                matrix[lo][lo + i], matrix[hi][hi - i] = matrix[hi][hi - i], matrix[lo][lo + i]
                matrix[lo][lo + i], matrix[hi - i][lo] = matrix[hi - i][lo], matrix[lo][lo + i]
            lo += 1
            hi -= 1


class Solution_85:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return 0

        dp = [[[0, 0] for _ in range(n + 1)] for _ in range(m + 1)]
        res = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i - 1][j - 1] == '1':
                    dp[i][j][0] = max(dp[i][j - 1][0], dp[i - 1][j - 1][0]) + 1
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if matrix[i - 1][j - 1] == '1':
                    dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][1]) + 1
                    res = max(res, dp[i][j][0] * dp[i][j][1])
        for ls in dp:
            print([tuple(t) for t in ls])
        return res


# print(Solution_85().maximalRectangle([["1","0","1","0","0"],
#                                       ["1","0","1","0","1"],
#                                       ["1","1","1","1","1"],
#                                       ["1","0","0","1","0"]]))

import sys

sys.stdout.flush()


class Solution_124:
    def maxPathSum(self, root: TreeNode) -> int:
        res = -float('inf')

        def post(root):
            nonlocal res
            if not root: return 0
            # root.val
            left = post(root.left)
            right = post(root.right)
            res = max(res, left + root.val + right, left + root.val, right + root.val, root.val)
            return max(max(left, right) + root.val, root.val)

        post(root)
        return int(res)

    def run(self, ls):
        def deserialize(data):
            if len(data) == 0:
                return None
            nodes = [TreeNode(int(x)) if x else x for x in data]
            kids = nodes[::-1]
            root = kids.pop()
            for node in nodes:
                if node:
                    if kids: node.left = kids.pop()
                    if kids: node.right = kids.pop()
            return root

        root = deserialize(ls)
        return self.maxPathSum(root)

    @staticmethod
    def debug():
        s = Solution_124()
        null = None
        assert s.run([2, -1]) == 2
        assert s.run([-1, 2]) == 2
        assert s.run([-10, 9, 20, null, null, 15, 7]) == 42
        assert s.run([1, 2, 3]) == 6
        assert s.run([8, 9, -6, null, null, 5, 9]) == 20
        assert s.run([-3]) == -3
        assert s.run([2, -1, -2]) == 2
        assert s.run([-1, 4, -2, null, null, 4, null]) == 5
        assert s.run([9, 6, -3, null, null, -6, 2, null, null, 2, null, -6, -6, -6]) == 16


class Solution_442:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        nums.sort()  # 排序 O(nlogn)
        res = []
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                res.append(nums[i])
        return res

    def findDuplicates(self, nums: List[int]) -> List[int]:
        # 使用正负表示是否访问过
        # 当前位置的数值等于位置号，则已经见过
        res = []
        i = 0
        while i < len(nums):
            if nums[i] > 0:  # 未见过
                t = nums[i]
                if nums[t - 1] < 0 and nums[t - 1] + t == 0:
                    res.append(t)
                    nums[i] = -t
                    i += 1
                else:
                    nums[i] = nums[t - 1]
                    nums[t - 1] = -t
            else:
                i += 1
        return res

    def findDuplicates(self, nums):
        # 先交换到自己的位置上，再查找一遍
        # 出现两次的数一定有一个不在自己的位置上
        for i in range(len(nums)):
            while i + 1 != nums[i]:
                t = nums[i]
                if nums[t - 1] == t:
                    break
                nums[i] = nums[t - 1]
                nums[t - 1] = t

        res = []
        for i in range(len(nums)):
            if i + 1 != nums[i]:
                res.append(nums[i])
        return res


class Solution_1208:
    def _TLE_equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        n = len(s)
        diff = [abs(ord(s[i]) - ord(t[i])) for i in range(n)]
        res = 0
        for i in range(n):
            cost = 0
            j = i
            while j < n:
                if cost + diff[j] <= maxCost:
                    cost += diff[j]
                else:
                    break
                j += 1
            res = max(res, j - i)
        return res

    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        # sliding window AC
        n = len(s)
        diff = [abs(ord(s[i]) - ord(t[i])) for i in range(n)]
        # print(diff)
        lo, hi = 0, 0
        s = 0
        res = 0
        while lo <= hi < n:
            while hi < n and s + diff[hi] <= maxCost:
                s += diff[hi]
                hi += 1
            res = max(res, hi - lo)
            if lo < hi:
                s -= diff[lo]
                lo += 1
            else:
                s = 0
                lo += 1
                hi += 1

        return res

    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        # more explicit sliding window
        n = len(s)
        diff = [abs(ord(s[i]) - ord(t[i])) for i in range(n)]
        s = lo = hi = 0
        res = 0
        while hi < n:
            s += diff[hi]
            while lo < hi and s > maxCost:
                s -= diff[lo]
                lo += 1
            if hi - lo + 1 > res and s <= maxCost:
                res = hi - lo + 1
            hi += 1
        return res


class Solution_1202:
    def _smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        # 当作图问题， 找出联通分量，对联通分量直接排序
        n = len(s)  # 图节点数
        ls = list(s)
        graph = {i: [] for i in range(n)}

        for a, b in pairs:
            graph[a].append(b)
            graph[b].append(a)

        visit = [0] * n

        def dfs(v, comp):
            visit[v] = 1
            comp.append(v)
            for nv in graph[v]:
                if visit[nv] == 0:
                    dfs(nv, comp)

        for i in range(n):
            if visit[i] == 0:
                comp = []
                dfs(i, comp)
                sc = sorted(comp, key=lambda x: s[x])
                comp.sort()
                # print(comp, sc)
                for k, x in zip(comp, sc):
                    ls[k] = s[x]
        return ''.join(ls)

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        # union find
        n = len(s)
        ls = list(s)
        ufs = [-1] * n

        def find(x):
            if ufs[x] != -1:
                r = find(ufs[x])
                ufs[x] = r
                return r
            return x

        for a, b in pairs:
            a, b = min(a, b), max(a, b)
            ra = a
            while ufs[ra] != -1:
                ra = ufs[ra]
            rb = b
            while ufs[rb] != -1:
                rb = ufs[rb]
            if rb != ra:
                ufs[rb] = ra
            while ufs[a] != -1:
                ufs[a], a = ra, ufs[a]
            while ufs[b] != -1:
                ufs[b], b = ra, ufs[b]

        comps = {}
        for i in range(n):
            if ufs[i] == -1:
                comps[i] = [i]

        for i in range(n):
            if ufs[i] != -1:
                x = i
                while ufs[x] != -1:
                    x = ufs[x]
                comps[x].append(i)

        for _, comp in comps.items():
            comp.sort()
            sc = sorted(comp, key=lambda x: s[x])
            for o, u in zip(comp, sc):
                ls[o] = s[u]
        return ''.join(ls)


class Solution_1169:
    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        # sliding window
        trans = [t.split(',') for t in transactions]
        for ls in trans:
            ls[1] = int(ls[1])
            ls[2] = int(ls[2])

        trans.sort(key=lambda ls: ls[1])
        res = set()
        lo = hi = 0
        while hi < len(trans):
            name, time, amount, city = trans[hi]
            if amount > 1000:
                res.add(f'{name},{time},{amount},{city}')

            while lo < hi and trans[hi][1] - trans[lo][1] > 60:
                lo += 1
            flag = False
            for i in range(lo, hi):
                if trans[i][0] == name and trans[i][3] != city:
                    flag = True  # res.add(f'{name},{time},{amount},{city}')
                    res.add(f'{trans[i][0]},{trans[i][1]},{trans[i][2]},{trans[i][3]}')
            if flag:
                res.add(f'{name},{time},{amount},{city}')
            hi += 1
        return list(res)


class Solution_1109:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        # 暴力解 TLE
        seats = [0] * n
        for i, j, k in bookings:
            for idx in range(i - 1, j):
                seats[idx] += k
        return seats

    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        #  difference array, 从线段树/树状数组的思想而来？
        # 树状数组的数据结构，它能在O(logn)内对数组的值进行修改和查询某一段数值的和
        # TODO 理解一下树状数组/线段树，看懂这个
        seats = [0] * n
        for i, j, k in bookings:
            seats[i - 1] += k
            if j < n:
                seats[j] -= k
        print(seats)
        for i in range(1, n):
            seats[i] += seats[i - 1]
        return seats


class Solution_1031:
    def maxSumTwoNoOverlap(self, A: List[int], L: int, M: int) -> int:
        prefix = [A[0]]
        for i in range(1, len(A)):
            prefix.append(prefix[-1] + A[i])
        prefix.append(0)
        res = 0
        for i in range(L - 1, len(A)):
            ml = prefix[i] - prefix[i - L]
            mm = 0
            for j in range(M - 1, i - L):
                tmm = prefix[j] - prefix[j - M]
                mm = max(mm, tmm)

            for j in range(i + 1, len(A) - M + 1):
                tmm = prefix[j + M - 1] - prefix[j - 1]
                mm = max(mm, tmm)

            res = max(res, ml + mm)
        return res


class Solution_1014:
    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        # can't do it by brute force, definitely TLE
        # 考虑固定i 搜索合适的 j
        # 这样问题变成 固定A[i] + i,搜索满足条件j>i 并且 A[j] - j最大。
        # 适合后缀处理
        suffix = [0] * len(A)
        suffix[-1] = len(A) - 1
        tj = len(A) - 1
        for j in range(len(A) - 2, -1, -1):
            if A[j] - j > A[tj] - tj:
                tj = j
            suffix[j] = tj
        suffix = suffix[1:] + suffix[:1]
        # print([A[j]-j for j in range(len(A))])
        # print(suffix)
        res = 0
        for i in range(len(A)):
            tj = suffix[i]
            if tj > i:
                tmp = A[i] + i + A[tj] - tj
                res = max(res, tmp)
        return res

    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        # can't do it by brute force, definitely TLE
        # 考虑固定i 搜索合适的 j
        # 这样问题变成 固定A[i] + i,搜索满足条件j>i 并且 A[j] - j最大。
        # 前缀处理
        prefix = [float('inf')]
        ti = 0
        for i in range(1, len(A)):
            prefix.append(ti)
            if A[i] + i > A[ti] + ti:
                ti = i
        # print([A[i]+i for i in range(len(A))])
        # print(prefix)
        res = 0
        for j in range(len(A)):
            ti = prefix[j]
            if ti < j:
                tmp = A[ti] + ti + A[j] - j
                res = max(res, tmp)
        return res


class Solution_1052:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
        # definitely need O(nlogn), even O(n)
        # 查找长度为X的和最大的子串？
        # 注意有些minute本身就是not grumpy。所以先将这些加到结果里，并把customer数置零
        res = 0
        for i, (c, g) in enumerate(zip(customers, grumpy)):
            if g == 0:
                res += c
                customers[i] = 0
        prefix = [customers[0]]
        for i in range(1, len(customers)):
            prefix.append(prefix[-1] + customers[i])
        prefix.append(0)
        mx = 0
        for i in range(len(customers) - X + 1):
            t = prefix[i + X - 1] - prefix[i - 1]
            mx = max(mx, t)
        return res + mx


class Solution_279:
    dp = [0]
    _dp = [0]

    def _numSquares(self, n):
        # print(len(self.dp))
        # 使用全局的dp数据来应对多次查询
        while len(self.dp) <= n:
            t = float('inf')
            i = 1
            while i * i <= len(self.dp):
                t = min(t, self.dp[len(self.dp) - i * i] + 1)
                i += 1
            self.dp.append(t)
        return self.dp[n]

    def _numSquares(self, n):
        # 和上面同样的思路，但是要比每次生成i*i<len(self.dp)快一点
        if len(self._dp) <= n:
            squres = [x * x for x in range(1, int(math.sqrt(n)) + 1)]
            for i in range(len(self._dp), n + 1):
                self._dp.append(1 + min(self._dp[i - s] for s in squres if i >= s))
        return self._dp[n]

    def _numSquares(self, n: int) -> int:
        # 直接dp生成，多次查询会超时
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            j = 0
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        # print(dp)
        return dp[n]

    def numSquares(self, n: int) -> int:
        # 数论方法。朗格朗日四平方和定理
        # 任意一个【正整数】可以写成四个【整数】的和
        # 因此本题的结果只可能有1，2，3，4(根据上述定理，最大是4)
        t = int(math.sqrt(n))
        if t * t == n:
            return 1
        # 根据四平方和定理，如果n能够被写成4^k*(8*m+7)的形式，结果为4（我也没看懂）
        while n % 4 == 0:
            n >>= 2
        if n % 8 == 7:
            return 4

        t = int(math.sqrt(n))  # n = 8*m + 7的形式
        for i in range(1, int(math.sqrt(n)) + 1):
            t = int(math.sqrt(n - i * i))
            if t * t == n - i * i:
                return 2
        return 3


######################################################################################################################
# 堆


class MedianFinder_295:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 使用两个堆
        self.small = []
        self.big = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.big, num)
        heapq.heappush(self.small, -heapq.heappop(self.big))
        if len(self.big) < len(self.small):
            heapq.heappush(self.big, -heapq.heappop(self.small))

    def findMedian(self) -> float:
        return (self.big[0] - self.small[0]) / 2 if len(self.small) == len(self.big) else self.big[0]


######################################################################################################################
# 拓扑排序问题

class Solution_207:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 拓扑排序，使用map和set降低复杂度，愚蠢的直接方法。
        n = numCourses
        edges = {i: set() for i in range(n)}
        # 有向边
        for a, b in prerequisites:
            edges[b].add(a)

        to_use = []
        unvisit = set(list(range(n)))
        while len(unvisit):
            for j in unvisit:
                cnt = len(edges[j])
                if cnt == 0:
                    to_use.append(j)
            if not to_use:
                return False
            while to_use:
                t = to_use.pop()
                unvisit -= {t}
                for j in range(n):
                    if t in edges[j]:
                        edges[j] -= {t}
        return True

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # topological sort via DFS
        n = numCourses
        edges = collections.defaultdict(list)
        for a, b in prerequisites:
            edges[a].append(b)
        visit = [-1] * n
        stack = []
        flag = True

        def dfs(v):
            nonlocal stack, visit, flag
            visit[v] = 0
            for i in edges[v]:
                if visit[i] == 0:
                    flag = False
                if visit[i] == -1:
                    dfs(i)
            visit[v] = 1
            stack.append(v)  # 回溯之前将节点压栈（此时已经访问过该节点指向的所有节点。

        for i in range(n):
            if visit[i] == -1:
                dfs(i)

        return flag  # flag表示是否有环（使用dfs染色法）


class Solution_210:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # topological sort via DFS
        n = numCourses
        edges = collections.defaultdict(list)
        for a, b in prerequisites:
            edges[a].append(b)  # 本来是b->a, 反过来存这样最后stack的结果就不用reverse
        visit = [-1] * n
        stack = []
        flag = True

        def dfs(v):
            nonlocal stack, visit, flag
            visit[v] = 0
            for i in edges[v]:
                if visit[i] == 0:
                    flag = False
                if visit[i] == -1:
                    dfs(i)
            visit[v] = 1
            stack.append(v)  # 回溯之前将节点压栈（此时已经访问过该节点指向的所有节点。

        for i in range(n):
            if visit[i] == -1:
                dfs(i)

        return stack if flag else []


######################################################################################################################


class Solution_215:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        t = heapq.nlargest(k, nums)
        return t[-1]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        kl = []
        for x in nums:
            heapq.heappush(kl, x)
            if len(kl) > k:
                heapq.heappop(kl)
        return kl[0]


class Solution_143:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return
        t = p = q = head
        while p:
            t = q
            q = q.next
            p = p.next
            if p:
                p = p.next
        t.next = None
        mid = ListNode(None)
        mid.next = q
        q = mid
        p = q
        while p:
            t = p.next
            p.next = q
            q = p
            p = t
        p = head
        while p and q != mid:
            t = q.next
            q.next = p.next
            p.next = q
            p = q.next
            q = t


class Solution_241:
    def diffWaysToCompute(self, inputs: str) -> List[int]:
        exp = []
        prev_idx = 0
        for i, c in enumerate(inputs):
            if c in ['*', '+', '-']:
                exp.append(int(inputs[prev_idx:i]))
                exp.append(c)
                prev_idx = i + 1
        exp.append(int(inputs[prev_idx:]))

        res = set()

        # 结果得数目是与运算符个数有关的卡特兰数
        # ===>中缀树
        def dfs(stack, exp, idx):
            while exp:
                op = exp.pop(0)
                if op == '+':
                    pass
                elif op == '-':
                    pass
                elif op == '*':
                    pass
                else:
                    stack.append(op)

        t = []

        def foo(exp, idx):
            nonlocal res, t

            for i in range(idx, len(exp)):
                op = exp[i]
                if op == '+':
                    foo(exp[:i - 1] + [exp[i - 1] + exp[i + 1]] + exp[i + 2:], i)
                elif op == '-':
                    foo(exp[:i - 1] + [exp[i - 1] - exp[i + 1]] + exp[i + 2:], i)
                elif op == '*':
                    foo(exp[:i - 1] + [exp[i - 1] * exp[i + 1]] + exp[i + 2:], i)
            for i in range(idx - 1, -1, -1):
                op = exp[i]
                if op == '+':
                    exp = exp[:i - 1] + [exp[i - 1] + exp[i + 1]]
                elif op == '-':
                    exp = exp[:i - 1] + [exp[i - 1] - exp[i + 1]]
                elif op == '*':
                    exp = exp[:i - 1] + [exp[i - 1] * exp[i + 1]]
            if len(exp) == 1:
                t.append(exp[0])

        foo(exp, 0)
        return t


class Solution_958:
    # determine if root is a complete binary tree
    def isCompleteTree(self, root):
        # 层序遍历，我们遇到第一个None之后的所有节点
        # 都应该是None, 如果有非None就不是完全二叉树。
        bfs = [root]
        i = 0
        while bfs[i]:
            bfs.append(bfs[i].left)
            bfs.append(bfs[i].right)
            i += 1
        return not any(bfs[i:])

    def isCompleteTree(self, root: TreeNode) -> bool:
        nodes = [(root, 1)]
        m = 1
        num = 0
        while num < len(nodes):
            node, v = nodes[num]
            m = max(m, v)
            num += 1
            if node.left:
                nodes.append((node.left, 2 * v))
            if node.right:
                nodes.append((node.right, 2 * v + 1))
        return num == m


class Solution_450:
    def deleteNode(self, root, key):
        def delete(root, key):
            if not root:
                return None
            if root.val > key:
                root.left = delete(root.left, key)
            elif root.val < key:
                root.right = delete(root.right, key)
            else:  # root.val == key:
                if not root.left:
                    return root.right
                elif not root.right:
                    return root.left

                rsmall = root.right
                while rsmall.left:
                    rsmall = rsmall.left
                root.val = rsmall.val
                root.right = delete(root.right, rsmall.val)
            return root

        return delete(root, key)

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        def delete(root, key):
            if not root:
                return None
            if root.val == key:  # 找到了要删除的节点
                # 叶子节点/只有一个子树的情况，提升另一个子树
                if not root.left:  # 没有左孩子（包括是叶子节点的情况)
                    return root.right
                if not root.right:
                    return root.left
                # 两个子树都存在，找右边的最小值
                rsmall = root.right
                while rsmall.left:
                    rsmall = rsmall.left
                # trick, 把要删除节点的左子树放到，右子树的最小值左边，因为右子树的最小值
                # 是一路向左查找到的，所以其原本左子树应该为空。相比于交换数据并把右子树的
                # 最小值删除，这是更快速的做法（但会大幅打乱原树的结构）
                rsmall.left = root.left
                return root.right
            else:
                if root.val > key:
                    # 递归，在左子树上删除
                    root.left = delete(root.left, key)
                elif root.val < key:
                    # 递归，在右子树上删除
                    root.right = delete(root.right, key)
                return root

        return delete(root, key)

    def deserialize(self, data):
        if len(data) == 0:
            return None
        data = data.replace(' ', '')
        nodes = [None if x == 'null' else TreeNode(int(x))
                 for x in data.split(',')]
        kids = nodes[::-1]
        root = kids.pop()
        for node in nodes:
            if node:
                if kids: node.left = kids.pop()
                if kids: node.right = kids.pop()
        return root


class Solution_343:
    def integerBreak(self, n: int) -> int:
        # 将一个数分成至少2个数的和，求将可能划分得最大积
        # 首先考虑将一个数n划分为两个数，显然有(n//2)(n-n//2)得积最大。
        # 即，划分应该尽可能得“相等”才能使得积最大。

        # 实际上，假如我们有将n划分为n/x个实数，然后最后得积分为x^(n/x)，对其求导
        # 有f'(x) = n * x^(n/x-2) * (1 - ln(x)) >>>>>>  幂指函数得求导法则
        # 然后发现当 0<x<e f'(x) > 0;  f(x)单调增
        #        当 x > e f'(x) < 0;  f(x)单调减
        # 因此取x = e得时候有最大值。
        # 回到这个题目上，因子只能取正整数，因此我们取接近e得数即 2<e<3。而且3更接近e。

        # 一个更直观得想法是 对于n >= 4, 我们有2(n-2) >= n。即对于大于4得数我们总能将其划分得到更大得积，因此可用的因子
        # 为1，2，3，显然1不在考虑之列，然后由6的划分 3 * 3 > 2 * 2 * 2，我们应该首选3。事实上因子2的个数最多不会超过2个。
        # 如果有3个2我们总可以用2个3替换它。
        if n == 2:
            return 1
        elif n == 3:
            return 2
        n3, c = divmod(n, 3)
        if c == 1:
            n3 -= 1
            c = 4
        elif c == 0:
            c = 1
        return pow(3, n3) * c

    @staticmethod
    def debug(self):
        s = Solution_343()
        res = []
        for i in range(2, 59):
            res.append(s.integerBreak(i))
        print(res)


class Solution_1247:
    def minimumSwap(self, s: str, t: str):
        n = len(s)
        x = y = 0
        for i in range(n):
            if s[i] != t[i]:
                if s[i] == 'x':
                    x += 1
                else:
                    y += 1

        if (x + y) & 1:  # 不等的有奇数个，无法通过交换实现相等
            return -1
        # x//2 ==> 有多少 xx 对（交换与顺序无关）
        # y//2 ==> 有多少 yy 对
        # xy / yx # 有剩余的x和y(有剩余 x 一定有剩余的y, 否则就是奇数个不等了)
        ret = x // 2 + y // 2 + 2 * (x & 1)

        return ret


import bisect


class Solution_1248:
    def numberOfSubarrays(self, A, k):
        n = len(A)
        s = [0] * (n + 1)
        for i in range(1, n + 1):
            s[i] = s[i - 1] + A[i - 1] % 2
        ret = 0
        for i in range(n):
            ret += bisect.bisect_right(s, s[i] + k) - bisect.bisect_left(s, s[i] + k)
        return ret


class Solution_1249:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        ls = list(s)
        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            elif c == ')':
                if stack:
                    stack.pop()
                else:
                    ls[i] = ''

        while stack:
            ls[stack.pop()] = ''

        return ''.join(ls)


class Solution_1250:
    def isGoodArray(self, nums: List[int]) -> bool:
        # 更相减损法（辗转相减），所有数的最大公约数为1肯定可以通过乘系数变成1
        a = nums[0]
        for i in range(1, len(nums)):
            a = math.gcd(a, nums[i])
        return a == 1


class WordDictionary_211:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        p = self.trie
        for c in word:
            if c not in p:
                p[c] = {}
            p = p[c]
        p['#'] = word

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """

        def foo(p, w):
            for i, c in enumerate(w):
                if c == '.':
                    for tc in p:
                        if tc != '#' and foo(p[tc], w[i + 1:]):
                            return True
                    return False
                elif c not in p:
                    return False
                else:
                    p = p[c]
            return '#' in p

        return foo(self.trie, word)

    @staticmethod
    def run():
        obj = WordDictionary_211()
        cmd = ["WordDictionary", "addWord", "addWord", "search", "search", "search", "search", "search", "search"]
        arg = [[], ["a"], ["a"], ["."], ["a"], ["aa"], ["a"], [".a"], ["a."]]
        res = []
        for c, a in zip(cmd, arg):
            print(c, a)
            if c == 'addWord':
                obj.addWord(a[0])
                res.append(None)
            elif c == 'search':
                t = obj.search(a[0])
                res.append(t)
            else:
                res.append(None)
        print(res)


class NumArray:
    # NumArray_307
    # BIT(树形数组)
    # stack exchange这个讲的特别好
    # https://cs.stackexchange.com/questions/10538/bit-what-is-the-intuition-behind-a-binary-indexed-tree-and-how-was-it-thought-a
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.nums = [0] + nums
        self.st = [0] * (self.n + 1)

        def init(x, val):
            while x <= self.n:
                self.st[x] += val
                x += self.lowbit(x)

        for i in range(1, self.n + 1):
            init(i, self.nums[i])

        print(self.st)

    def update(self, i: int, val: int) -> None:
        i += 1
        diff = val - self.nums[i]
        self.nums[i] = val
        while i <= self.n:
            self.st[i] += diff
            i += self.lowbit(i)
        print(self.st)

    def sumRange(self, i: int, j: int) -> int:
        i += 1
        j += 1

        def sm(x):
            res = 0
            while x >= 1:
                res += self.st[x]
                x -= self.lowbit(x)
            return res

        # print(sm(j), sm(i-1))
        return sm(j) - sm(i - 1)

    @staticmethod
    def lowbit(x):
        return x & (-x)

    @staticmethod
    def run():
        obj = None
        cmd = ["NumArray", "update", "update", "update", "sumRange", "update", "sumRange", "update", "sumRange",
               "sumRange", "update"]
        arg = [[[7, 2, 7, 2, 0]], [4, 6], [0, 2], [0, 9], [4, 4], [3, 8], [0, 4], [4, 1], [0, 3], [0, 4], [0, 4]]
        res = []
        for c, a in zip(cmd, arg):
            if c == 'sumRange':
                t = obj.sumRange(a[0], a[1])
                res.append(t)
            elif c == 'update':
                obj.update(a[0], a[1])
                res.append(None)
            else:
                obj = NumArray(a[0])
                res.append(None)
        print(res)


class Solution_328:
    # 链表
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        dummy = ListNode(None)
        tail = dummy
        q = p = head
        n = 0
        while p:
            n += 1
            if n & 1 == 0:
                q.next = p.next
                p.next = None
                tail.next = p
                tail = tail.next
                p = q.next
            else:
                q = p
                p = p.next
        q.next = dummy.next
        return head

    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        # 使用第一个作为奇数头节点，第二个作为偶数头节点，尾插
        odd, even, even_head = head, head.next, head.next
        while even and even.next:
            odd.next = even.next
            odd = odd.next

            even.next = odd.next
            even = even.next
        odd.next = even_head
        return head




class Twitter:
    # this get AC, but was terribuly written
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.tweets = collections.defaultdict(list)
        self.follows = {}
        self.timestampe = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        self.tweets[userId].append((self.timestampe, tweetId))
        self.timestampe += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        if userId not in self.follows:
            return [tt[1] for tt in self.tweets[userId][-10:]][::-1]

        targets = []
        if len(self.tweets[userId]) > 0:
            targets += [userId]
        for f in self.follows[userId]:
            if len(self.tweets[f]) > 0:
                targets += [f]
        targets = list(set(targets))
        indices = [0] * len(targets)
        for i, t in enumerate(targets):
            indices[i] = len(self.tweets[t]) - 1

        res = []
        while len(res) < 10:
            candidates = [self.tweets[t][i] for t, i in zip(targets, indices)]
            if len(candidates) == 0:
                break
            m, idx = -1, -1
            for i, c in enumerate(candidates):
                if c[0] > m:
                    m = c[0]
                    idx = i
            if m != -1 and idx != -1:
                res.append(candidates[idx][1])
                indices[idx] -= 1
                if indices[idx] < 0:
                    indices.pop(idx)
                    targets.pop(idx)

        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId in self.follows:
            self.follows[followerId].add(followeeId)
        else:
            self.follows[followerId] = set([followeeId])

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId in self.follows and followeeId in self.follows[followerId]:
            self.follows[followerId].remove(followeeId)

    @staticmethod
    def debug():
        obj = Twitter()
        cmds = ["Twitter", "postTweet", "getNewsFeed", "follow", "getNewsFeed", "unfollow", "getNewsFeed"]
        args = [[], [1, 1], [1], [2, 1], [2], [2, 1], [2]]
        for cmd, arg in zip(cmds, args):
            if cmd == 'Twitter':
                pass
            elif cmd == "postTweet":
                obj.postTweet(*arg)
            elif cmd == "getNewsFeed":
                obj.getNewsFeed(*arg)
            elif cmd == 'follow':
                obj.follow(*arg)
            else:
                obj.unfollow(*arg)


class Solution_347:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        c = collections.Counter(nums)
        res = sorted([k for k in c], key=lambda k: c[k], reverse=True)
        return res[:k]


class Solution_294:
    def decodeString(self, s: str) -> str:
        # return ''
        le = s.find('[')
        if le == -1:  # no [ or ] return the string
            return s

        ki = 0
        while ki < le:
            if s[ki].isdigit():
                break
            ki += 1
        k = int(s[ki:le]) if s[ki:le] else 0
        tmp = s[:ki]
        cnt = 1
        # 找与[对应的]
        ri = len(s) - 1
        for i in range(le + 1, len(s)):
            if s[i] == '[':
                cnt += 1
            elif s[i] == ']':
                cnt -= 1
                if cnt == 0:
                    ri = i
                    break

        t = self.decodeString(s[le + 1:ri])
        return tmp + ''.join([t] * k) + self.decodeString(s[ri + 1:])

    def decodeString(self, s: str) -> str:
        # use stack
        stack = []
        res = []
        k = 0
        t = []
        for i, c in enumerate(s):
            if c.isdigit():
                k = k * 10 + int(c)
            elif c == '[':
                stack.append([k, t])
                k = 0
                t = []
            elif c == ']':
                n, tmp = stack.pop()
                t = tmp + t * n
                if not stack:
                    res.extend(t)
                    t = []
            else:
                t.append(c)

        if t:
            res.extend(t)
        return ''.join(res)


class Solution_726:
    def countOfAtoms(self, formula: str) -> str:

        def helper(s, hi):
            count = collections.defaultdict(int)
            lo = 0
            while lo < hi:
                name = s[lo]
                lo += 1
                while lo < hi and s[lo].islower():
                    name += s[lo]
                    lo += 1
                cnt = 0
                while lo < hi and s[lo].isdigit():
                    cnt = cnt * 10 + int(s[lo])
                    lo += 1
                count[name] += max(cnt, 1)
            return count

        def foo(formula):
            n = len(formula)
            left = formula.find('(')
            if left == -1:
                return helper(formula, n)

            cnt = 1
            ri = n - 1
            for i in range(left + 1, n):
                if formula[i] == '(':
                    cnt += 1
                elif formula[i] == ')':
                    cnt -= 1
                    if cnt == 0:
                        ri = i
                        break
            tmp = helper(formula, left)
            res = foo(formula[left + 1:ri])

            k, idx = 0, ri + 1
            while idx < n and formula[idx].isdigit():
                k = k * 10 + int(formula[idx])
                idx += 1
            for key in res:
                res[key] *= k

            remain = foo(formula[idx:])

            for k, v in remain.items():
                res[k] = res.get(k, 0) + v

            for k, v in tmp.items():
                res[k] = res.get(k, 0) + v

            return res

        res = foo(formula)
        tmp = sorted([(k, v) for k, v in res.items()])
        ans = []
        for t in tmp:
            if t[1] == 1:
                ans += [t[0]]
            else:
                ans += [t[0], str(t[1])]
        return ''.join(ans)


class Solution_406:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        n = len(people)
        res = [None] * n
        indices = list(range(n))
        d = {}
        for p in people:
            if p[0] in d:
                d[p[0]].append(p)
            else:
                d[p[0]] = [p]

        keys = sorted(d.keys())
        for k in keys:
            tmp = sorted(d[k], key=lambda x: x[1], reverse=True)
            for h, k in tmp:
                res[indices[k]] = [h, k]
                indices.pop(k)
        return res


import time


class Solution_739:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        # the key to slove this problem is that temperature's range falls in [30, 100].
        temps = {}
        for i, t in enumerate(T):
            if t in temps:
                temps[t].append(i)
            else:
                temps[t] = [i]
        # for k, v in temps.items():
        #     print(k,':', len(v))
        # return []
        res = []
        for i, t in enumerate(T):
            idx = len(T)
            for tmp in range(t + 1, 101):
                if tmp in temps:
                    ls = temps[tmp]
                    lo, hi = 0, len(ls)
                    while lo < hi:
                        mi = lo + ((hi - lo) >> 1)
                        if ls[mi] > i:
                            hi = mi
                        else:
                            lo = mi + 1
                    if lo < len(ls):
                        idx = min(idx, ls[lo] - i)
                        if idx == 1:
                            break
            res.append(idx if idx < len(T) else 0)
        return res

    def dailyTemperatures(self, T: List[int]) -> List[int]:
        n = len(T)
        m = 100 - 30 + 1
        temps = [[n] * n for _ in range(m)]
        print(len(temps), len(temps[0]))
        for i in range(30, 101):
            if T[n - 1] == i:
                temps[i - 30][n - 1] = n - 1
            for j in range(n - 2, -1, -1):
                if T[j] == i:
                    temps[i - 30][j] = j
                else:
                    temps[i - 30][j] = temps[i - 30][j + 1]
        res = []
        for i, t in enumerate(T):
            idx = n
            for tmp in range(t + 1, 101):
                idx = min(temps[tmp - 30][i], idx)

            res.append(idx - i if idx < n else 0)
        return res


class Solution_332:
    # Graph, 欧拉路径
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        edge = sorted(tickets)
        visit = [0] * len(edge)
        res = []

        def dfs(path):
            nonlocal res
            if res:
                return
            if all(x == 1 for x in visit):
                res = path[:]
                return
            # we waste a lot time loop here, as we finished visit an edge, we can cut it.
            for i, t in enumerate(edge):
                if visit[i] == 0 and t[0] == path[-1]:
                    visit[i] = 1
                    dfs(path + [t[1]])
                    visit[i] = 0

        dfs(['JFK'])
        return res

    def _findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # this is the method to find a Eulerian path.
        # * 在欧拉路径问题中，必须有start节点（这个题目中是'JFK'）和end节点，且start和end节点的度数目为奇数。
        # * start节点和end节点可以是同一个节点（其他节点全是偶数度）
        # * 所以先找到end节点，然后delete end节点后回溯
        m = collections.defaultdict(list)
        e = sorted(tickets, reverse=True)
        for a, b in e:
            m[a] += b,  # append(b) += [b]

        res = []

        def dfs(cur):
            nonlocal res
            while m[cur]:
                dfs(m[cur].pop())
            res.append(cur)  # 第一个入栈是 end 节点

        dfs('JFK')
        res.reverse()
        return res

    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # 迭代的算法
        m = collections.defaultdict(list)
        e = sorted(tickets, reverse=True)
        for a, b in e:
            m[a] += b,  # append(b) += [b]

        res = []
        stack = ['JFK']
        while stack:
            while m[stack[-1]]:
                stack.append(m[stack[-1]].pop())
            res.append(stack.pop())
        res.reverse()
        return res

class Solution_131:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def foo(idx, path):
            nonlocal res
            if idx == 0:
                res.append(path[::-1])
            for i in range(idx-1, -1, -1):
                t = s[i:idx]
                if t == t[::-1]:
                    foo(i, path+[t])
        foo(len(s), [])
        return res



class Solution_560:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # if all the number are positive, we can calculate the prefix sum.
        # then we can use binary search, but in this problem, there are negative numbers.
        # NOTE: continuous(subarray)
        prefix = [0]
        for x in nums:
            prefix.append(prefix[-1] + x)

        cnt = 0
        for i in range(len(prefix)):
            for j in range(i + 1, len(prefix)):
                if prefix[j] - prefix[i] == k:
                    cnt += 1
        return cnt

    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = 0
        for i in range(len(nums)):
            s = 0
            for j in range(i, len(nums)):
                s += nums[j]
                if s == k:
                    cnt += 1
        return cnt

    def subarraySum(self, nums, k):
        # O(n) solution, hash map
        cnt = 0
        s = 0
        memo = {0: 1}
        for i in range(len(nums)):
            s += nums[i]
            if s - k in memo:
                cnt += memo[s - k]
            memo[s] = memo.get(s, 0) + 1
        return cnt


class Solution_827:
    def largestIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])  # 1 <= m,n <= 50
        island = [[0] * n for _ in range(m)]

        def dfs(x, y, num):
            nonlocal island
            island[x][y] = num
            cnt = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and island[nx][ny] == 0 and grid[nx][ny] == 1:
                    cnt += dfs(nx, ny, num)
            return cnt

        zeros = set()
        size = {}
        no = 1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and island[i][j] == 0:
                    size[no] = dfs(i, j, no)
                    no += 1
                elif grid[i][j] == 0:
                    zeros.add((i, j))
        res = max(size.values()) if size else 0  # 有可能 grid 全为0
        for i, j in zeros:
            lands = set()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and island[ni][nj] != 0:
                    lands.add(island[ni][nj])

            res = max(res, 1 + sum(size[i] for i in lands))
        return res


class Solution_300:
    # 这个题目Python卡时间太严了，O(n^2)也会 TLE
    def lengthOfLIS(self, nums: List[int]) -> int:
        # O(n^2) dp
        # 0-1背包的变形题，枚举（递归） --> 带memo的递归 --> dp --> dp优化

        def foo(idx, path):
            # 这种前向和后向混在仪器，而且状态不明确，不好转换为带memo的搜索
            print('#', idx, path)
            if idx == len(nums):
                return len(path) - 1
            r = 0
            if nums[idx] > path[-1]:
                r = foo(idx + 1, path + [nums[idx]])
            return max(r, foo(idx + 1, path))

        def helper(last, idx):
            if idx == len(nums):
                print('#', last, idx, '0')
                return 0
            r = 0
            if last < 0 or nums[last] < nums[idx]:
                r = 1 + helper(idx, idx + 1)
            t = max(r, helper(last, idx + 1))
            print('#', last, idx, t)
            return t

        return helper(-1, 0)

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 暴力转为带memo的搜索
        memo = {}

        def helper(last, idx):
            nonlocal memo
            if idx == len(nums):
                return 0

            if (last, idx) in memo:
                return memo[(last, idx)]

            r = 0
            if last < 0 or nums[last] < nums[idx]:
                r = 1 + helper(idx, idx + 1)
            memo[(last, idx)] = max(r, helper(last, idx + 1))
            return memo[(last, idx)]

        return helper(-1, 0)

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 带memo的搜索 转换为 后向DP
        # 这里i 对应last, j对应idx. 存的时候因为last有-1， 所以+1存储
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        max_len = 0
        for last in range(n - 1, -2, -1):
            for idx in range(n - 1, -1, -1):
                r = 0
                if last < 0 or nums[last] < nums[idx]:
                    r = 1 + dp[idx + 1][idx]
                dp[last + 1][idx] = max(r, dp[last + 1][idx + 1])
                max_len = max(dp[last + 1][idx], max_len)
        for ls in dp:
            print(ls)
        return max_len

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 优化空间，前向DP, 跟上面的DP过程不太一样了
        # dp[i] = max(dp[j]) + 1,  0 <= j < i
        # ans = max(dp)

        n = len(nums)
        if n == 0:
            return 0  # 边界条件
        dp = [0] * n
        dp[0] = 1
        max_len = 0
        for i in range(n):
            tmp = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    tmp = max(tmp, dp[j])

            dp[i] = tmp + 1
            max_len = max(max_len, dp[i])
        print(dp)
        return max_len

    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp + binary search
        # O(n logn)
        n = len(nums)
        dp = [0] * n
        length = 0
        for x in nums:
            idx = bisect.bisect_left(dp, x, 0, length)
            dp[idx] = x
            if idx == length:
                length += 1

        return length

    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp + binary search的另一种写法
        # 这里使用 dp 和 path 数组, 以及length 能够重建出其中一个最长子序列
        # 例如：[3, 4, -1, 5, 8, 2, 3, 12, 7, 9, 10]
        # 对应的dp: [2, 5, 6, 8, 9, 10, 0, 0, 0, 0, 0]
        # 对应的path: [-1, 0, -1, 1, 3, 2, 5, 4, 6, 8, 9]
        # length为：6
        # 首先 dp[length - 1] = 10
        # path[10] = 9
        # path[9] = 8
        # path[8] = 6
        # path[6] = 5
        # path[5] = 2
        # path[2] = -1
        # 重建的最长子序列为
        # index   2   5   6   8   9   10
        #  num   -1   2   3   7   9   10
        n = len(nums)
        tmp = [0] * n
        dp = [0] * n
        path = [-1] * n
        length = 0
        for i, x in enumerate(nums):
            idx = bisect.bisect_left(tmp, x, 0, length)
            dp[idx] = i
            tmp[idx] = x
            if idx > 0:
                path[i] = dp[idx - 1]
            if idx == length:
                length += 1
        print(dp)
        print(path)
        return length


class Solution_673:
    # TODO
    def findNumberOfLIS(self, nums: List[int]) -> int:
        # dp[i] = max(dp[j]) + 1,  0 <= j < i
        # ans = max(dp)

        n = len(nums)
        if n == 0:
            return 0  # 边界条件
        memo = collections.defaultdict(int)
        dp = [0] * n
        dp[0] = 1
        max_len = 0
        for i in range(n):
            tmp = 0
            cnt = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] > tmp:
                        tmp = dp[j]
                        cnt = 1
                    elif dp[j] == tmp:
                        cnt += 1

            dp[i] = tmp + 1
            print('#', i, dp[i], cnt)
            memo[dp[i]] += max(memo.get(tmp, 1), 1)
            max_len = max(max_len, dp[i])
        print(dp)
        print(memo)
        print(max_len)
        return memo[max_len]


# print(Solution_673().findNumberOfLIS([3, 4, -1, 5, 8, 2, 3, 12, 7, 9, 10]))
class Solution_943:
    def shortestSuperstring(self, A: List[str]) -> str:
        # 一个例子：
        # ["catg","ctaagt","gcta","ttca","atgcatc"]
        # gctaagttcatgcatc
        # gcta
        #  ctaagt
        #       ttca
        #         catg
        #          atgcatc
        # 显然这应该是与两个字符串的前后缀相同部分的长度有关
        pass




