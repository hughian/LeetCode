from typing import List
import collections
import itertools
import functools
import math
import heapq
import string


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


class Solution_386:
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

    def lexicalOrder(self, n: int):
        cur = 1
        res = []
        for i in range(1, n + 1):
            res.append(cur)
            if cur * 10 <= n:  # 向下一层
                cur *= 10
            elif cur % 10 != 9 and cur + 1 <= n:  # 右兄弟，向右
                cur += 1
            else:
                while (cur // 10) % 10 == 9:  # 既不能向下，也不能向右，就向上回溯
                    cur //= 10

                cur = cur // 10 + 1  # 回溯节点向右
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


class Solution_310:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        adj = collections.defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        if n == 1:
            return [0]

        que = collections.deque()
        for k in adj:
            if len(adj[k]) == 1:
                que.append(k)

        res = []

        # 从叶节点开始 BFS，最后剩下的节点 1/2 个，就是答案。
        while que:
            L = len(que)
            res = list(que)
            for _ in range(L):
                node = que.popleft()
                for v in adj[node]:
                    adj[v].remove(node)
                    if len(adj[v]) == 1:
                        que.append(v)
        return res


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
        # DP方法TLE
        dp = [float('inf')] * len(nums)
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(1, nums[i] + 1):
                if i + j < len(nums):
                    dp[i + j] = min(dp[i + j], dp[i] + 1)

        return int(dp[-1])

    def jump(self, nums: List[int]) -> int:
        # jump game I中的greedy思路，将其看作一个BFS问题，类似于到最后一个节点的最短路
        # 如[2, 3,1,1,4]可以转换为
        #         2         lv0
        #        / \
        #       3   1       lv1
        #      / |  |
        #     1  4  1       lv2
        if len(nums) < 2:  # 边界要单独处理
            return 0
        pos = 0
        lv = 0
        i = 0
        while i <= pos:
            lv += 1
            next_pos = pos
            while i <= pos:  # BFS 访问当前层, 相当于这一层的全部入队列
                next_pos = max(next_pos, nums[i] + i)
                if next_pos >= len(nums) - 1:
                    return lv
                i += 1
            pos = next_pos

        return 0


class Solution_1306:
    def canReach(self, arr: List[int], start: int) -> bool:
        # BFS 处理，从 start 向两端扩展，遇到第一个 arr[i] = 0 就是结果
        # 注意不能出界

        que = collections.deque([start])
        visit = [0] * len(arr)
        while que:
            t = que.popleft()
            if arr[t] == 0:
                return True
            if t - arr[t] >= 0 and visit[t - arr[t]] == 0:
                que.append(t - arr[t])
                visit[t - arr[t]] = 1
            if t + arr[t] < len(arr) and visit[t + arr[t]] == 0:
                que.append(t + arr[t])
                visit[t + arr[t]] = 1
        return False


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
        # difference array, 对于要更新一个区间内的每一个数，
        # 不使用暴力的方法更新每一个数，而是使用首位标记的方法，
        # 在区间头标记一个正值，区间结尾后一个标记一个负值，这
        # 样最后查询值时从前向后加和之后给出的出结果。
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
            edges[a].append(b)  # 本来是b->din, 反过来存这样最后stack的结果就不用reverse
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


class Solution_980:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        todo = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] != -1:
                    todo += 1
                if grid[i][j] == 1:
                    start = (i, j)
                elif grid[i][j] == 2:
                    end = (i, j)

        ans = 0

        def dfs(i, j, todo):
            nonlocal ans
            todo -= 1
            if todo < 0: return
            if (i, j) == end:
                if todo == 0:
                    ans += 1
                return
            grid[i][j] = -1
            for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] % 2 == 0:  # 2 也要算进去
                    dfs(x, y, todo)
            grid[i][j] = 0

        dfs(start[0], start[1], todo)
        return ans

    def uniquePathsIII(self, grid: List[List[int]]) -> int:

        m, n = len(grid), len(grid[0])
        code = lambda x, y: 1 << (x * n + y)
        target = 0  # 最长只有 20 位
        for i in range(m):
            for j in range(n):
                if grid[i][j] % 2 == 0:
                    target |= code(i, j)
                if grid[i][j] == 1:
                    start = (i, j)
                elif grid[i][j] == 2:
                    end = (i, j)

        @functools.lru_cache(None)
        def dfs(i, j, todo):
            if (i, j) == end:
                return int(todo == 0)
            ans = 0
            for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] % 2 == 0:  # 2 也要算进去
                    if todo & code(x, y):
                        ans += dfs(x, y, todo ^ code(x, y))
            return ans

        return dfs(start[0], start[1], target)


class Solution_958:
    # determine if root is din complete binary tree
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


class Solution_526:
    def countArrangement(self, N: int) -> int:
        # 暴力， TLE
        res = 0
        for ls in itertools.permutations(list(range(1, N + 1))):
            if all(x % (i + 1) == 0 or (i + 1) % x == 0 for i, x in enumerate(ls)):
                res += 1
        return res

    def countArrangement(self, N: int) -> int:
        # dfs + 剪枝
        def dfs(i, A):
            if i >= N:
                return 1 if (A[N] % i == 0 or i % A[N] == 0) else 0
            res = 0
            for j in range(i, N + 1):
                if i % A[j] == 0 or A[j] % i == 0:
                    tmp = A[:]
                    tmp[i], tmp[j] = tmp[j], tmp[i]
                    res += dfs(i + 1, tmp)
            return res

        return dfs(1, [0]+list(range(1, N + 1)))

    def countArrangement(self, N: int) -> int:
        # using set
        def dfs(i, A):
            if i > N:
                return 1
            return sum(dfs(i + 1, A - {x}) for x in A if x % i == 0 or i % x == 0)

        return dfs(1, set(range(1, N + 1)))


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
        Adds din word into the data structure.
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
        arg = [[], ["din"], ["din"], ["."], ["din"], ["aa"], ["din"], [".din"], ["din."]]
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


class Solution_910:
    def smallestRangeII(self, A: List[int], K: int) -> int:
        # 找最小的可能值
        A.sort()
        mi, ma = A[0], A[-1]
        ans = ma - mi

        for i in range(len(A) - 1):
            a, b = A[i], A[i + 1]  # 我们不用考虑 a - K, b + K 这个肯定更大
            ans = min(ans, max(ma - K, a + K) - min(mi + K, b - K))

        return ans


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
        Compose din new tweet.
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
        Follower follows din followee. If the operation is invalid, it should be din no-op.
        """
        if followerId in self.follows:
            self.follows[followerId].add(followeeId)
        else:
            self.follows[followerId] = set([followeeId])

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows din followee. If the operation is invalid, it should be din no-op.
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


class Solution_1319:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        # 顶点 0 ~ n-1
        # 联通分量数 - 1
        # edge case, # edge < # node - 1
        if len(connections) < n - 1:
            return -1
        G = collections.defaultdict(list)
        for a, b in connections:
            G[a].append(b)
            G[b].append(a)

        visit = [0] * n

        def dfs(i):
            for v in G[i]:
                if visit[v] == 0:
                    visit[v] = 1
                    dfs(v)

        cnt = 0
        for i in range(n):
            if visit[i] == 0:
                cnt += 1
                visit[i] = 1
                dfs(i)
        return cnt - 1

    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        # union find
        if len(connections) < n - 1:
            return -1

        uf = list(range(n))

        def find(x):
            if uf[x] == x:
                return x
            uf[x] = find(uf[x])
            return uf[x]

        for a, b in connections:
            fa = find(a)
            fb = find(b)
            if fa != fb:
                uf[fa] = fb

        cnt = sum(i == find(i) for i in range(n))
        return cnt - 1


class Solution_1106:
    # stack
    def parseBoolExpr(self, expression: str) -> bool:
        stack = []
        for c in expression:
            if c == ')':
                tt = []
                while stack and stack[-1] != '(':
                    ch = stack.pop()
                    tt.append(ch)
                stack.pop()
                op = stack.pop()
                if op == '&':
                    res = all(ch == 't' for ch in tt)
                elif op == '|':
                    res = any(ch == 't' for ch in tt)
                elif op == '!':
                    res = False if tt[0] == 't' else True
                else:
                    raise
                stack.append('t' if res else 'f')
            elif c != ',':
                stack.append(c)
        return True if stack[0] == 't' else False


class Solution_394:
    def decodeString(self, s: str) -> str:
        # 使用正则表达式
        import re
        while '[' in s:
            s = re.sub(r'(\d+)\[([^\[^\]]*)\]', lambda m: int(m.group(1)) * m.group(2), s)
        return s

    def decodeString(self, s: str) -> str:
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

    def decodeString(self, s):
        stack = []
        for c in s:
            if c == ']':
                tmp = []
                while stack[-1] != '[':
                    tmp.append(stack.pop())
                tmp.reverse()
                stack.pop()  # '['
                base, cur = 1, 0
                while stack and stack[-1].isdigit():
                    cur += base * int(stack.pop())
                    base *= 10
                stack.append(''.join(tmp) * cur)
            else:
                stack.append(c)
        return ''.join(stack)


class Solution_1096:
    def braceExpansionII(self, exp):
        group = [[]]
        lv = 0
        for i, c in enumerate(exp):
            if c == '{':
                if lv == 0:  # 如果是最外层，记录这个 { 的开始
                    start = i + 1  # 注意跳过了 i('{')
                lv += 1
            elif c == '}':
                lv -= 1
                if lv == 0:  # 如果是最外层，递归处理内层
                    group[-1].append(self.braceExpansionII(exp[start:i]))
            elif c == ',' and lv == 0:
                group.append([])
            elif lv == 0:
                group[-1].append([c])  # 特殊 case R("din")
        print(group)
        ss = set()
        for g in group:
            print('#', g, *g)
            ss |= set(map(''.join, itertools.product(*g)))
        return sorted(ss)

    def braceExpansionII(self, exp):
        # 使用 res 保存一对 {} 内已经处理好的部分，cur 保存正在处理的部分。
        # 遇到 } 出栈时与前面的结果进行 product, 然后最终的结果保存在 res + cur 中
        # 这个思路太精妙了
        stack, res, cur = [], [], []
        for i, c in enumerate(exp):
            if c.isalpha():
                cur = [t + c for t in cur or ['']]
            elif c == '{':
                stack.append(res)
                stack.append(cur)
                res, cur = [], []
            elif c == '}':
                pre = stack.pop()
                pre_res = stack.pop()
                cur = [p + t for t in res + cur for p in pre or ['']]
                res = pre_res
            elif c == ',':
                res += cur
                cur = []
        return sorted(set(res + cur))


class Solution_1240:
    def tilingRectangle(self, n: int, m: int) -> int:
        # 如果有一个解的话，我们可以从下往上填充正方形，这样就每个位置我们存一个高度，就像 Skyline 一样
        # 一个数组，存着每个位置的高度（可以看作将大矩形分成许多1x1的小格子）
        res = m * n

        def foo(height, moves):
            nonlocal res
            if all(h == n for h in height):  # 全部填满了，就是填满了整个矩形
                res = min(res, moves)
                return
            if moves >= res:  # 如果填充数目已经超过了目前的最小结果，剪枝
                return

            min_h = min(height)  # 当前的最低高度
            idx = height.index(min_h)
            ridx = idx + 1
            while ridx < m and height[ridx] == min_h:
                ridx += 1
            # idx ~ ridx 都是 height[i] == min_h 的位置，==> 宽度
            # n - min_h => 高度。
            # 我们最大可以填充一个边长为 min(宽度， 高度) 的正方形
            for i in range(min(ridx - idx, n - min_h), 0, -1):
                tmp = height[:]
                for j in range(i):
                    tmp[idx + j] += i
                # 将这一段填充，之后继续递归
                foo(tmp, moves + 1)

        foo([0] * m, 0)
        return res


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
            # we waste din lot time loop here, as we finished visit an edge, we can cut it.
            for i, t in enumerate(edge):
                if visit[i] == 0 and t[0] == path[-1]:
                    visit[i] = 1
                    dfs(path + [t[1]])
                    visit[i] = 0

        dfs(['JFK'])
        return res

    def _findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # this is the method to find din Eulerian path.
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
            for i in range(idx - 1, -1, -1):
                t = s[i:idx]
                if t == t[::-1]:
                    foo(i, path + [t])

        foo(len(s), [])
        return res


class Solution_973:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # 大顶堆，存 K 个小的
        heap = []
        for a, b in points:
            if len(heap) < K:
                heapq.heappush(heap, [-(a ** 2 + b ** 2), a, b])
            else:
                if -(a ** 2 + b ** 2) > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, [-(a ** 2 + b ** 2), a, b])
        return [[a, b] for _, a, b in heap]


class Solution_1254:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        flag = True

        def dfs(i, j):
            nonlocal flag
            grid[i][j] = 2
            for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 0:
                    if x in {0, m - 1} or y in {0, n - 1}:
                        flag = False
                    dfs(x, y)

        res = 0
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if grid[i][j] == 0:
                    flag = True
                    dfs(i, j)
                    if flag:
                        res += 1
        return res


class Solution_658:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        if x <= arr[0]:
            return arr[:k]
        elif arr[-1] <= x:
            return arr[-k:]

        ix = bisect.bisect_left(arr, x)
        i = max(0, ix - k - 1)
        j = min(len(arr) - 1, ix + k - 1)

        while j - i + 1 > k:
            if i < 0 or abs(x - arr[i]) <= abs(arr[j] - x):
                j -= 1
            elif j > len(arr) - 1 or abs(x - arr[i]) > abs(x - arr[j]):
                i += 1
        return arr[i:j + 1]


class Solution_1344:
    def angleClock(self, hour: int, minutes: int) -> float:
        # 时针每 一个小时 走 30 deg, 每分钟 0.5 deg
        # 分针每 一分钟走 360 / 60 = 6 deg
        # 00:00/12:00 角度是0
        deg = abs(minutes * 6 - (hour % 12) * 30 - 0.5 * minutes)
        return deg if deg <= 180 else 360 - deg


class Solution_539:
    def findMinDifference(self, timePoints: List[str]) -> int:
        # 24 * 60 = 1440
        tmp = list(map(lambda s: 60 * int(s[:2]) + int(s[3:]), timePoints))
        tmp.sort()
        tmp.append(tmp[0] + 1440)
        m = float('inf')
        for i in range(1, len(tmp)):
            m = min(tmp[i] - tmp[i - 1], m)
        return m


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


class Solution_337:
    def rob(self, root: TreeNode) -> int:
        memo = {}

        def foo(root, rob_parent):
            nonlocal memo
            if not root: return 0
            if (root, rob_parent) in memo:
                return memo[root, rob_parent]
            if not rob_parent:
                left = max(foo(root.left, True), foo(root.left, False))
                right = max(foo(root.right, True), foo(root.right, False))
                r = left + right
            else:
                r = root.val + foo(root.left, False) + foo(root.right, False)
            memo[root, rob_parent] = r
            return r

        return max(foo(root, True), foo(root, False))

    def rob(self, root):
        # 多往下考虑一步，就不用写这么多 True False
        memo = {}

        def foo(root):
            nonlocal memo
            if not root: return 0
            if root in memo:
                return memo[root]
            val = 0
            if root.left:
                val += foo(root.left.left) + foo(root.left.right)

            if root.right:
                val += foo(root.right.left) + foo(root.right.right)

            memo[root] = max(val + root.val, root(root.left) + root(root.right))
            return memo[root]

        return foo(root)

    def rob(self, root):
        def foo(root):
            # 返回 不选root, 和选 root
            if not root: return 0, 0
            left = foo(root.left)
            right = foo(root.right)
            not_rob_root = max(left[0], left[1]) + max(right[0], right[1])
            rob_root = root.val + left[0] + right[0]
            return not_rob_root, rob_root

        return max(foo(root))


class Solution_1305:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        def foo(root, res):
            if root:
                foo(root.left, res)
                res.append(root.val)
                foo(root.right, res)

        res = []
        foo(root1, res)
        foo(root2, res)
        return sorted(res)

    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        # 直接通过树来合并, 使用 stack 非递归来解决这个问题
        def push_all_left(stk, node):
            while node:
                stk.append(node)
                node = node.left

        res = []
        s1, s2 = [], []
        push_all_left(s1, root1)
        push_all_left(s2, root2)

        while s1 or s2:
            if not s1:
                s = s2
            elif not s2:
                s = s1
            else:
                if s1[-1].val < s2[-1].val:
                    s = s1
                else:
                    s = s2

            p = s.pop()
            res.append(p.val)
            push_all_left(s, p.right)
        return res


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


class Solution_969:
    def pancakeSort(self, A: List[int]) -> List[int]:

        def reverse(A, idx):
            i, j = 0, idx - 1
            while i < j:
                A[i], A[j] = A[j], A[i]
                i += 1
                j -= 1

        res = []
        for k in range(len(A)):
            m, mi = float('-inf'), -1
            for i in range(len(A) - k):
                if A[i] > m:
                    m = A[i]
                    mi = i
            if mi == -1:
                break

            if mi != len(A) - k - 1:
                if mi > 0:
                    res.append(mi + 1)
                    res.append(len(A) - k)
                    reverse(A, mi + 1)
                    reverse(A, len(A) - k)
                else:
                    res.append(len(A) - k)
                    reverse(A, len(A) - k)
            # print(mi, A)
        return res

    def pancakeSort(self, A: List[int]) -> List[int]:
        # 相比于每次找最大值的位置的方法，我们可以一开始将位置记下来
        ans = []
        n = len(A)
        B = sorted(range(1, n + 1), key=lambda i: -A[i - 1])  # 将 A 值大下标排在前面
        for i in B:
            for f in ans:
                if i <= f:  # i < f 说明 i 在 f 翻转的范围内，下标被改变到对应位置
                    # 0 ... i ... f-1  (flip f 是从 0 ~ f-1)
                    i = f + 1 - i
            ans.extend([i, n])
            n -= 1
        return ans


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


class Solution_794:
    def winner(self, board):
        wins = set()
        ts = []
        for i in range(3):
            ts.append(board[i])

        for j in range(3):
            ts.append([board[0][j], board[1][j], board[2][j]])

        ts.append([board[0][0], board[1][1], board[2][2]])
        ts.append([board[0][2], board[1][1], board[2][0]])

        for t in ts:
            if all(c == 'x_train' for c in t):
                wins.add('x_train')
            elif all(c == 'O' for c in t):
                wins.add('O')
        return wins

    def validTicTacToe(self, board: List[str]) -> bool:
        if all(s == '   ' for s in board):
            return True
        x, o = 0, 0
        for s in board:
            for c in s:
                if c == 'x_train':
                    x += 1
                elif c == 'O':
                    o += 1
        wins = self.winner(board)
        if x == o and len(wins) < 2 and 'x_train' not in wins:
            return True
        elif x == o + 1 and len(wins) < 2 and 'O' not in wins:
            return True
        return False


class Solution_779:
    def kthGrammar(self, N: int, K: int) -> int:
        ks = []
        # 当作树，从叶子向上，左孩子为 0， 右孩子为 1
        for i in range(N):
            t = (K + 1)
            ks.append(t & 1)
            K = t // 2
        ks.reverse()  # reverse 为从根节点向下的过程
        # 向左保持不变，向右数字翻转
        ans = 0
        for x in ks:
            ans = x ^ ans
        return ans


class Solution_636:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ans = [0] * n
        stack = []
        for log in logs:
            _id, op, time = log.split(':')
            _id, time = int(_id), int(time)
            if op == 'start':
                if stack:
                    last_id, last_st_time = stack[-1]
                    ans[last_id] += time - last_st_time
                    stack[-1] = (last_id, time)
                stack.append((_id, time))
            else:
                last_id, last_st_time = stack.pop()
                ans[_id] += time - last_st_time + 1
                if stack:
                    stack[-1] = (stack[-1][0], time + 1)

        return ans


class Solution_735:
    # stack
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []

        for a in asteroids:
            append = True
            while stack and stack[-1] > 0 and a < 0:
                absa = abs(a)
                absb = abs(stack[-1])
                if absa >= absb:
                    stack.pop()

                if absa <= absb:
                    append = False
                    break
            if append:
                stack.append(a)
        return stack


class Solution_856:
    # stack 括号匹配
    def scoreOfParentheses(self, S: str) -> int:
        stack = []
        ans = 0
        tmp = 0.5
        for j, c in enumerate(S):
            if c == '(':
                stack.append(j)
                tmp *= 2
            else:
                i = stack.pop()
                if j == i + 1:
                    ans += int(tmp)
                tmp *= 0.5
        return ans


class Solution_1003:
    def isValid(self, s: str) -> bool:
        # 注意 aabcbabcc 也是 valid 的，根据定义，[din]abc[bc] 是 valid , 我们可以把它分成两部分 x_train = aabcb, Y = c
        # 于是有 [aabcb]abc[c] 也是 valid
        t = ''
        while s != t:
            s, t = s.replace('abc', ''), s
        return s == ''

    def isValid(self, s: str) -> bool:
        # 这也是个括号匹配类的问题，一个 c 匹配一对 din,b
        stack = []
        for c in s:
            if c == 'c':
                if stack[-2:] != ['din', 'b']:
                    return False
                stack.pop()
                stack.pop()
            else:
                stack.append(c)
        return len(stack) == 0


class Solution_921:
    # stack, 括号匹配
    def _minAddToMakeValid(self, S: str) -> int:
        stack = []
        ans = 0
        for c in S:
            if c == '(':
                stack.append('(')
            else:
                if stack:
                    stack.pop()
                else:
                    ans += 1
        return ans + len(stack)

    def minAddToMakeValid(self, S: str) -> int:
        left = 0
        ans = 0
        for c in S:
            if c == '(':
                left += 1
            else:
                if left > 0:
                    left -= 1
                else:
                    ans += 1
        return ans + left


class Solution_946:
    # Stack, 模拟出入栈的过程
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        n = len(pushed)
        i = j = 0
        while i < n and j < n:
            if pushed[i] != popped[j]:
                while stack and stack[-1] == popped[j]:
                    stack.pop()
                    j += 1
                stack.append(pushed[i])
                i += 1
            else:
                i += 1
                j += 1
        while stack and stack[-1] == popped[j]:
            stack.pop()
            j += 1
        return i == n and j == n


# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
class NestedInteger:
    def __init__(self, value=None):
        if value:
            self.value = value
        else:
            self.value = []

    def __repr__(self):
        return f'{self.value}'

    def isInteger(self) -> bool:
        """
        @return True if this NestedInteger holds din single integer, rather than din nested list.
        """
        return isinstance(self.value, int)

    def add(self, elem):
        """
        Set this NestedInteger to hold din nested list and adds din nested integer elem to it.
        :rtype void
        """
        if self.isInteger():
            self.value = [self.value]
        self.value.append(elem)

    def getInteger(self) -> int:
        """
        @return the single integer that this NestedInteger holds, if it holds din single integer
        Return None if this NestedInteger holds din nested list
        """
        if self.isInteger():
            return self.value
        return None

    def getList(self):
        """
        @return the nested list that this NestedInteger holds, if it holds din nested list
        Return None if this NestedInteger holds din single integer
        """
        if not self.isInteger():
            return self.value
        return None


class NestedIterator:
    # 3,4,1
    def __init__(self, nestedList: [NestedInteger]):
        def foo(nint):
            if nint.isInteger():
                yield nint.getInteger()
            else:
                for nn in nint.getList():
                    for ns in foo(nn):
                        yield ns

        def bar(ls):
            for nl in ls:
                for nn in foo(nl):
                    yield nn

        self.nl = bar(nestedList)

    def next(self) -> int:
        t = self.peek
        self.peek = None
        return t

    def hasNext(self) -> bool:
        try:
            self.peek = next(self.nl)
        except Exception as e:
            print(e)
            self.peek = None
        finally:
            return self.peek is not None


class Solution_385:
    # stack, 括号匹配，递归
    def deserialize(self, s: str) -> NestedInteger:
        if s[0] == '[':
            stack = []
            last = 1
            res = NestedInteger()
            for i in range(len(s)):
                if s[i] == '[':
                    stack.append(i)
                elif s[i] == ']':
                    idx = stack.pop()
                    if len(stack) == 1:
                        a = self.deserialize(s[idx:i + 1])
                        res.add(a)
                    elif len(stack) == 0 and len(s[last:i]) > 0:
                        n = int(s[last:i])
                        res.add(NestedInteger(n))
                    last = i + 1
                elif s[i] == ',':
                    if len(stack) == 1 and i > last:
                        n = int(s[last:i])
                        res.add(NestedInteger(n))
                    last = i + 1
            return res
        else:
            return NestedInteger(int(s))


class Solution_1209:
    def removeDuplicates(self, s: str, k: int) -> str:
        stack = []
        for c in s:
            if stack and stack[-1][0] == c:
                stack[-1][1] += 1
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append([c, 1])
        res = []
        for c, n in stack:
            res += [c] * n
        return ''.join(res)


class Solution_1190:
    def reverseParentheses(self, s: str) -> str:
        def reverse(ls, i, j):
            while i < j:
                ls[i], ls[j] = ls[j], ls[i]
                i += 1
                j -= 1

        ls = list(s)
        stack = []
        for i, c in enumerate(ls):
            if c == '(':
                stack.append(i)
            elif c == ')':
                idx = stack.pop()
                reverse(ls, idx, i)
                ls[idx] = ls[i] = ''
        return ''.join(ls)


class Solution_1124:
    def longestWPI(self, hours: List[int]) -> int:
        nums = [1 if h > 8 else -1 for h in hours]
        prefix = [0]
        for x in nums:
            prefix.append(prefix[-1] + x)
        m = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                if prefix[j + 1] - prefix[i] > 0:
                    m = max(m, j + 1 - i)
        return m


class Solution_1104:
    def pathInZigZagTree(self, label: int) -> List[int]:
        res = [label]
        lv = 1 + int(math.log2(label))
        while label > 1:
            if lv & 1:
                ori = (2 ** lv - 1) - (label - (2 ** (lv - 1)))

            else:
                ori = 2 ** (lv - 1) - (label - (2 ** lv - 1))
            label = ori // 2
            res.append(label)
            lv -= 1
        res.reverse()
        return res


class Solution_1041:
    def isRobotBounded(self, instructions: str) -> bool:
        pos = (0, 0)
        dirt = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        di = 0
        for i in instructions:
            if i == 'G':
                pos = (pos[0] + dirt[di][0], pos[1] + dirt[di][1])
            elif i == 'L':
                di = (di + 1) % 4
            else:
                di = (di - 1 + 4) % 4
        # pos 是一组指令执行完后的移动向量，di 是朝向
        # 如果一组指令执行后，回到原点，即 pos == (0, 0) 显然有 Bound
        # 如果移动向量不为零，而此时与最开始朝向（朝北）不同，则有Bound
        if pos == (0, 0) or di != 0:
            return True
        return False


class Solution_1017:
    def baseNeg2(self, N: int) -> str:
        if N == 0:
            return '0'
        res = []
        while N != 0:
            N, r = divmod(N, -2)  # 对于 -2 余数只可能是 0 / -1,
            # 如果余数为 -1 代表 N // (-2) 进行了向下取整，我们要改成向上取整，即通过除过的商 +1.
            # TODO
            if r < 0:
                r = 1  # -1 -> 1
                N += 1
            res.append(str(r))
        res.reverse()
        return ''.join(res)

    def baseNeg2(self, N: int) -> str:
        if N == 0:
            return '0'
        res = []
        n = -N
        while n != 0:
            n, r = divmod(-n, 2)
            res.append(str(r))

        res.reverse()
        return ''.join(res)

    def baseNeg2(self, N: int) -> str:
        if N == 0:
            return '0'
        res = []
        while N != 0:
            res.append(str(N & 1))
            N = -(N // 2)

        res.reverse()
        return ''.join(res)


class Solution_1073:
    def addNegabinary(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # 变成数字再算
        def to_num(arr):
            i = len(arr) - 1
            base = 1
            res = 0
            while i >= 0:
                res += base * arr[i]
                base *= -2
                i -= 1
            return res

        def to_arr(num):
            # -2 进制, 用了第 1017 题的代码
            if num == 0:
                return [0]
            res = []
            while num != 0:
                num, r = divmod(num, -2)
                if r < 0:
                    r = 1
                    num += 1
                res.append(r)
            while len(res) > 1 and res[-1] == 0:
                res.pop()
            res.reverse()
            return res

        ans = to_num(arr1) + to_num(arr2)
        return to_arr(ans)

    def addNegabinary(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # 直接计算，唯一的区别是 减去进位
        # 因为第 i 位和第 i + 1 位总是符号相反的。
        i, j = len(arr1) - 1, len(arr2) - 1
        c = 0
        res = []
        while i >= 0 and j >= 0:
            c, s = divmod(arr1[i] + arr2[j] - c, 2)
            res.append(s)
            i -= 1
            j -= 1

        while i >= 0:
            c, s = divmod(arr1[i] - c, 2)
            res.append(s)
            i -= 1
        while j >= 0:
            c, s = divmod(arr2[j] - c, 2)
            res.append(s)
            j -= 1
        while c != 0:
            c, s = divmod(-c, 2)
            res.append(s)
        while len(res) > 1 and res[-1] == 0:
            res.pop()
        res.reverse()
        return res


class Solution_1015:
    def smallestRepunitDivByK(self, K: int) -> int:
        n = 1
        cnt = 1
        while n % K and cnt < K:
            n %= K
            n = n * 10 + 1
            cnt += 1
        return -1 if n % K else cnt


class Solution_963:
    def minAreaFreeRect(self, points: List[List[int]]) -> float:
        eps = 1e-8
        res = float('inf')
        points = set(map(tuple, points))
        for a, b, c in itertools.permutations(points, 3):
            d = b[0] + c[0] - a[0], b[1] + c[1] - a[1]
            if d in points:
                ab = complex(b[0] - a[0], b[1] - a[1])  # 使用复数表示向量
                ac = complex(c[0] - a[0], c[1] - a[1])
                if abs(ab.real * ac.real + ab.imag * ac.imag) < eps:  # 点积等于0,垂直
                    area = abs(ab) * abs(ac)  # abs(complex)是取模长，长 x 宽
                    res = min(res, area)
        return 0 if res == float('inf') else res

    def minAreaFreeRect(self, points: List[List[int]]) -> float:
        points = [complex(*p) for p in points]
        seen = collections.defaultdict(list)
        for P, Q in itertools.combinations(points, 2):
            center = (P + Q) / 2
            radius = abs(center - P)
            seen[center, radius].append(P)

        res = float('inf')
        for (c, r), candidates in seen.items():
            for P, Q in itertools.combinations(candidates, 2):
                res = min(res, abs(P - Q) * abs(P - (2 * c - Q)))
        return res if res < float('inf') else 0


class Solution_164:
    def maximumGap(self, nums: List[int]) -> int:
        # using sort, O(n log n)
        nums = sorted(nums)
        res = 0
        # print(nums)
        for a, b in zip(nums[1:], nums[:-1]):
            res = max(res, a - b)
        return res

    def maximumGap(self, nums: List[int]) -> int:
        # Radix Sort 基数排序, O(d·(n+k)) ~ O(n)
        if len(nums) < 2:
            return 0

        mval = max(nums)
        e = 1
        radix = 10  # base 10
        aux = [0] * len(nums)
        while mval // e > 0:
            count = [0] * radix
            # 计数
            for i in range(len(nums)):
                count[(nums[i] // e) % 10] += 1
            #
            for i in range(1, len(count)):
                count[i] += count[i - 1]

            for i in range(len(nums) - 1, -1, -1):
                count[(nums[i] // e) % 10] -= 1
                aux[count[(nums[i] // e) % 10]] = nums[i]

            for i in range(len(nums)):
                nums[i] = aux[i]

            e *= 10

        res = 0
        for i in range(len(nums) - 1):
            res = max(res, nums[i + 1] - nums[i])
        return res

    def maximumGap(self, nums: List[int]) -> int:
        # 桶排序 + 鸽笼原理
        # 首先，最大的 gap 一定是满足 >=  t = (max - min) / (n-1)
        #       n 个数有 n-1 个 gap。假设有一个间距为 t 的等差数列，
        #       gap 值都是 t, maxGap 自然也是 t, 假如你想减小某两个元素
        #       之间的 gap, 那么相邻的 gap 自然会增加，导致 maxGap 增大
        # 其次，我们可以选择一个桶的大小（capacity）为 b，其中 b 满足 1 < b <= t，这样
        #       每个桶内任意两个元素的 gap 小于 t, 都不会最终的答案
        # 于是，我们只需要比较两个桶之间的元素。
        if len(nums) < 2:
            return 0

        mini, maxi = min(nums), max(nums)
        bucket_size = max(1, (maxi - mini) // (len(nums) - 1))
        bucket_num = (maxi - mini) // bucket_size + 1

        buckets = [[False, float('inf'), float('-inf')] for _ in range(bucket_num)]
        for num in nums:
            idx = (num - mini) // bucket_size
            buckets[idx][0] = True
            buckets[idx][1] = min(num, buckets[idx][1])
            buckets[idx][2] = max(num, buckets[idx][2])

        prev_bucket_max = mini
        res = 0
        for bucket in buckets:
            if bucket[0]:  # 只算用到的
                res = max(res, bucket[1] - prev_bucket_max)
                prev_bucket_max = bucket[2]
        return res


class Solution_127:
    # word Ladder
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # BFS
        # 这个问题可以转换为无向图问题，单词表示图节点，两个单词差一个字母表示两点间有一条无向边
        # 这样寻找最短的变换过程就是找一个开始节点 beginWord 和终止节点 endWord 之间的最短路问题。
        # 寻找一个节点到另一个节点的最短路可以使用 BFS 来解决，
        alpha = string.ascii_lowercase

        def next_words(w, words):
            for i in range(len(w)):
                for c in alpha:
                    nw = w[:i] + c + w[i + 1:]
                    if nw in words:
                        yield nw

        que = collections.deque()
        que.append([beginWord])
        wordList = set(wordList)
        lv = 1
        visited = set()
        while que:
            L = len(que)
            lv += 1
            for _ in range(L):
                path = que.popleft()
                for w in next_words(path[-1], wordList):
                    visited.add(w)
                    path.append(w)
                    if w == endWord:
                        # print(path)
                        return lv
                    else:
                        que.append(path[:])
                    path.pop()
            for w in visited:
                if w in wordList:
                    wordList.remove(w)
            visited.clear()

        return 0

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        alpha = string.ascii_lowercase

        def next_words(w, words):
            for i in range(len(w)):
                for c in alpha:
                    nw = w[:i] + c + w[i + 1:]
                    if nw in words:
                        yield nw

        que = collections.deque()
        que.append(beginWord)
        wordList = set(wordList)
        lv = 1
        visited = set()
        while que:
            L = len(que)
            lv += 1
            for _ in range(L):
                word = que.popleft()
                for nw in next_words(word, wordList):
                    visited.add(nw)
                    if nw == endWord:
                        return lv
                    else:
                        que.append(nw)
            for w in visited:
                if w in wordList: wordList.remove(w)
            visited.clear()
        return 0

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # bi BFS
        # 假如每个节点的相邻节点是固定的，则单项的 BFS 搜索的节点数目随着搜索的层数指数级增加，
        # 使用双向 BFS 可以大量减少搜索的节点数目，begin 和 end 的层数都很低，因此搜索节点数目会大量减少。

        words = set(wordList)
        if endWord not in words:
            return 0
        else:
            words.remove(endWord)

        if beginWord in words:
            words.remove(beginWord)

        alpha = 'abcdefghijklmnopqrstuvwxyz'
        begin_set, end_set = {beginWord}, {endWord}
        lv = 1
        while begin_set and end_set:
            if len(begin_set) < len(end_set):
                is_begin_small, small, big = True, begin_set, end_set
            else:
                is_begin_small, small, big = False, end_set, begin_set

            next_lv = set()
            lv += 1
            for word in small:
                for i in range(len(word)):
                    for ch in alpha:
                        nw = word[:i] + ch + word[i + 1:]
                        if nw in big or nw in words:
                            if nw in big:
                                return lv
                            next_lv.add(nw)

            for word in next_lv:
                words.remove(word)
            if is_begin_small:
                begin_set = next_lv
            else:
                end_set = next_lv

        return 0


class Solution_126:
    def _findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        # 使用标准 BFS 的流程
        res = []
        que = collections.deque()
        que.append([beginWord])
        word_list = set(wordList)
        lv = 1
        min_lv = float('inf')
        visited = set()
        alpha = 'abcdefghijklmnopqrstuvwxyz'
        while que:
            # print(que)
            path = que.popleft()
            if len(path) > lv:
                for w in visited:
                    word_list.remove(w)
                visited.clear()
                if len(path) > min_lv:
                    break
                else:
                    lv = len(path)
            last_word = list(path[-1])

            for i, wc in enumerate(last_word):
                new_word = last_word[:]
                for c in alpha:
                    new_word[i] = c
                    tmp = ''.join(new_word)
                    if tmp in word_list:
                        path.append(tmp)
                        visited.add(tmp)
                        if tmp == endWord:
                            min_lv = lv
                            res.append(path[:])
                        else:
                            que.append(path[:])
                        path.pop()
        return res

    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        # BFS 过程，队列中存储 path, 而不是最后的 word
        res = []
        que = collections.deque()
        que.append([beginWord])
        word_list = set(wordList)
        visited = set()
        alpha = 'abcdefghijklmnopqrstuvwxyz'
        while que:
            print(que)
            L = len(que)
            for _ in range(L):
                path = que.popleft()
                last_word = list(path[-1])

                for i, wc in enumerate(last_word):
                    new_word = last_word[:]
                    for c in alpha:
                        new_word[i] = c
                        tmp = ''.join(new_word)
                        if tmp in word_list:
                            path.append(tmp)
                            visited.add(tmp)  # endWord 也只会被访问一次
                            if tmp == endWord:
                                res.append(path[:])  # 遇到 endWord 队列中就没有加入新元素了
                            else:
                                que.append(path[:])
                            path.pop()

            for w in visited:
                word_list.remove(w)
            if endWord in visited:
                break
            visited.clear()

        return res

    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        # two-end BFS, 根据 begin_set 和 end_set 两个集合的长度来决定扩展哪个集合
        # 在剩余的所有 word 上循环，更新路径映射，最后根据路径映射建立结果
        words = set(wordList)
        if endWord not in words:
            return []
        else:
            words.remove(endWord)

        if beginWord in words:
            words.remove(beginWord)

        alpha = 'abcdefghijklmnopqrstuvwxyz'
        begin_set, end_set = {beginWord}, {endWord}
        ans_found = False
        mp = collections.defaultdict(list)

        while begin_set and end_set:
            if len(begin_set) < len(end_set):
                is_begin_small, small, big = True, begin_set, end_set
            else:
                is_begin_small, small, big = False, end_set, begin_set

            next_lv = set()
            for word in small:
                for i in range(len(word)):
                    for ch in alpha:
                        nw = word[:i] + ch + word[i + 1:]
                        if nw in big or nw in words:
                            if nw in big:
                                ans_found = True
                            next_lv.add(nw)

                            if is_begin_small:
                                mp[word].append(nw)
                            else:
                                mp[nw].append(word)
            if ans_found:
                break

            for word in next_lv:
                words.remove(word)
            if is_begin_small:
                begin_set = next_lv
            else:
                end_set = next_lv

        if not ans_found:
            return []

        res = [[beginWord]]
        while res[0][-1] != endWord:
            tmp = []
            for path in res:
                for word in mp[path[-1]]:
                    tmp.append(path + [word])
            res = tmp[:]
        return res


class Solution_329:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # DFS + Memo
        dirt = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        memo = [[-1] * n for _ in range(m)]

        def dfs(i, j, matrix, visited):
            nonlocal memo
            if memo[i][j] != -1:
                return memo[i][j]
            visited[i][j] = 1
            r = 1
            for dx, dy in dirt:
                nx, ny = i + dx, j + dy
                if 0 <= nx < m and 0 <= ny < n and visited[nx][ny] == 0 and matrix[nx][ny] > matrix[i][j]:
                    r = max(r, 1 + dfs(nx, ny, matrix, visited))
            visited[i][j] = 0
            memo[i][j] = r
            return r

        res = 0
        for i in range(m):
            for j in range(n):
                if memo[i][j] == -1:
                    res = max(res, dfs(i, j, matrix, [[0] * n for _ in range(m)]))
                else:
                    res = max(res, memo[i][j])
        return res

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # DFS + Memo
        matrix = {i + j * 1j: val for i, row in enumerate(matrix) for j, val in enumerate(row)}
        memo = {k: -1 for k in matrix.keys()}

        def dfs(pos, matrix, visited):
            nonlocal memo
            if memo[pos] != -1:
                return memo[pos]
            visited.add(pos)
            r = 1
            for npos in (pos + 1, pos - 1, pos + 1j, pos - 1j):
                if npos in matrix and npos not in visited and matrix[npos] > matrix[pos]:
                    r = max(r, 1 + dfs(npos, matrix, visited))

            visited.remove(pos)
            memo[pos] = r
            return r

        res = 0
        for pos in memo.keys():
            if memo[pos] == -1:
                res = max(res, dfs(pos, matrix, set()))
            else:
                res = max(res, memo[pos])
        return res

    def _longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        matrix = {i + j * 1j: val for i, row in enumerate(matrix) for j, val in enumerate(row)}
        print(matrix)
        print(sorted(matrix, key=matrix.get))
        length = {}
        for z in sorted(matrix, key=matrix.get):
            length[z] = 1 + max([length[ttt] for ttt in (z + 1, z - 1, z + 1j, z - 1j) if
                                 ttt in matrix and matrix[z] > matrix[ttt]] or [0])
        return max(length.values() or [0])

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # 使用复数表示二维坐标
        matrix = {i + j * 1j: val for i, row in enumerate(matrix) for j, val in enumerate(row)}
        dp = {k: 0 for k in matrix.keys()}

        def dfs(pos):
            if not dp[pos]:
                dp[pos] = 1 + max(
                    dfs(pos - 1) if pos - 1 in matrix and matrix[pos] > matrix[pos - 1] else 0,
                    dfs(pos + 1) if pos + 1 in matrix and matrix[pos] > matrix[pos + 1] else 0,
                    dfs(pos - 1j) if pos - 1j in matrix and matrix[pos] > matrix[pos - 1j] else 0,
                    dfs(pos + 1j) if pos + 1j in matrix and matrix[pos] > matrix[pos + 1j] else 0,
                )
            return dp[pos]

        return max([dfs(pos) for pos in matrix.keys()] or [0])


class Solution_1349:
    def maxStudents(self, seats: List[List[str]]) -> int:
        # bitmasking DP
        def _count(x):
            if x <= 0:
                return 0
            cnt = 0
            while x:
                x = x & (x - 1)
                cnt += 1
            return cnt

        m, n = len(seats), len(seats[0])
        # 将数组变成 bitmap
        validity = []
        for i in range(m):
            cur = 0
            for j in range(n):
                cur = (cur << 1) + (seats[i][j] == '.')
            validity.append(cur)

        N = 1 << n
        dp = [[-1] * N for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            valid = validity[i - 1]
            for j in range(N):
                # 所有的可选集合中如果是 valid 的子集 且 j 中没有相邻的两个元素
                if (j & valid) == j and not (j & (j >> 1)):
                    for k in range(N):
                        # 与 j 对应的没有左前/右前的 k, 且 dp[i-1][k] 有效
                        if not (j & (k >> 1)) and not (j & (k << 1)) and dp[i - 1][k] != -1:
                            # 这一行新加的数就是对应 j 中有效值（1的个数）
                            dp[i][j] = max(dp[i][j], dp[i - 1][k] + _count(j))

        return max(dp[-1])


class Solution_1345:
    def minJumps(self, arr: List[int]) -> int:
        # 最小步数，BFS
        n = len(arr)
        if n < 2:
            return 0

        memo = collections.defaultdict(list)
        for i, x in enumerate(arr):
            memo[x].append(i)

        que = collections.deque([0])
        visit = [0] * len(arr)
        visit[0] = 1
        lv = 0
        while que:
            L = len(que)
            for _ in range(L):
                t = que.popleft()
                if t == n - 1:
                    return lv

                ls = memo[arr[t]]
                ls.extend([t - 1, t + 1])

                for nt in ls:
                    if 0 <= nt < n and visit[nt] == 0:
                        que.append(nt)
                        visit[nt] = 1
                ls.clear()
            lv += 1
        return 0


class Solution_1316:
    def distinctEchoSubstrings(self, text: str) -> int:
        # 简单思路是枚举所有的非空子串 O(n^2)
        n = len(text)
        res = set()
        for i in range(n):
            for L in range(2, n - i + 1, 2):
                mid = i + L // 2
                left = text[i:mid]
                right = text[mid:i + L]
                if left == right:
                    res.add(left)
        # print(res)
        return len(res)

    def distinctEchoSubstrings(self, text: str) -> int:
        # Rolling Hash
        # hash list 记录 s[0:i] 的 hash 值
        # pow list 存下 base ^ i
        # hash(s[i:j]) = hash[j] - hash[i] * base^(j-i), j>i
        # hash[i] = s[0]*base^(i-1) + s[1]*base^(i-2) +...+ s[i-1]*base^0
        # hash[j] = s[0]*base^(j-1) + s[1]*base^(j-2) +...+ s[i-1]*base^(j-i)  + ... + s[j-1]*base^0

        # hash[j] - hash[i] * base^(j-i)
        #    = s[0]*base^(j-1) + s[1]*base^(j-2) + ... + s[i-j]*base^(j-i) + ... + s[j-1]*base^0
        #      -(s[0]*base^(j-1) + s[1]*base^(j-2) + ... + s[i-1]*base^(j-i))
        #    = s[i]*base^(j-i) + ... + s[j-1] * base^0
        # 即是 hash(s[i:j])
        n = len(text)
        base, MOD = 29, int(1e9 + 7)
        Hash = [0] * (n + 1)
        Pow = [1] * (n + 1)
        for i in range(1, n + 1):
            Hash[i] = (Hash[i - 1] * base + ord(text[i - 1])) % MOD
            Pow[i] = Pow[i - 1] * base % MOD

        def get_hash(i, j):
            return (Hash[j] - Hash[i] * Pow[j - i] % MOD + MOD) % MOD

        res = set()
        # cnt = 0
        for i in range(n):
            for L in range(2, n - i + 1, 2):
                mid = i + L // 2
                hs1 = get_hash(i, mid)  # 应为求模运算的增加，时间并没有变快
                hs2 = get_hash(mid, i + L)
                if hs1 == hs2:  # 这里应该加上冲突判断 and s[i:mid] == s[mid:i+L]
                    res.add(hs1)
        return len(res)


class Solution_558:
    def intersect(self, quadTree1: 'Node', quadTree2: 'Node') -> 'Node':
        if not quadTree1:
            return quadTree2
        if not quadTree2:
            return quadTree1

        def copy(t1, t2):
            t1.topLeft = t2.topLeft
            t1.topRight = t2.topRight
            t1.bottomLeft = t2.bottomLeft
            t1.bottomRight = t2.bottomRight

        # Node 是一个四叉树节点
        Node = collections.namedtuple("Node", "val isLeaf topLeft topRight bottomLeft bottomRight")

        node = Node(False, False, None, None, None, None)
        if quadTree1.isLeaf and quadTree2.isLeaf:
            node.val = quadTree1.val or quadTree2
            node.isLeaf = True
            return node
        elif quadTree1.isLeaf:
            if quadTree1.val:
                node.val = True
                node.isLeaf = True
            else:
                copy(node, quadTree2)
                node.isLeaf = False
        elif quadTree2.isLeaf:
            if quadTree2.val:
                node.val = True
                node.isLeaf = True
            else:
                copy(node, quadTree1)
                node.isLeaf = False
        else:
            node.topLeft = self.intersect(quadTree1.topLeft, quadTree2.topLeft)
            node.topRight = self.intersect(quadTree1.topRight, quadTree2.topRight)
            node.bottomLeft = self.intersect(quadTree1.bottomLeft, quadTree2.bottomLeft)
            node.bottomRight = self.intersect(quadTree1.bottomRight, quadTree2.bottomRight)
            ls = [node.topLeft, node.topRight, node.bottomLeft, node.bottomRight]
            if all(x.isLeaf for x in ls):
                if all(x.val for x in ls) or all(not x.val for x in ls):
                    node.isLeaf = True
                    node.val = node.topLeft.val
                    node.topLeft = None
                    node.topRight = None
                    node.bottomLeft = None
                    node.bottomRight = None
        return node


class Solution_1284:
    def minFlips(self, mat: List[List[int]]) -> int:
        # 3 x 3 枚举就行了 最大 2^(3x3)
        m, n = len(mat), len(mat[0])

        def flip(i, j):
            mat[i][j] = 1 - mat[i][j]
            for ni, nj in [(i - 1, j), (i + 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= ni < m and 0 <= nj < n:
                    mat[ni][nj] = 1 - mat[ni][nj]

        res = float('inf')

        def foo(pos, mat, step):
            nonlocal res
            print(pos, mat, step)
            if pos >= m * n:
                a = [mat[i][j] for i in range(m) for j in range(n)]
                if all(x == 0 for x in a):
                    res = min(res, step)
                return

            foo(pos + 1, mat, step)
            i, j = divmod(pos, n)
            print(i, j)
            flip(i, j)
            foo(pos + 1, mat, step + 1)
            flip(i, j)

        foo(0, mat, 0)
        return res if res < float('inf') else -1


class Solution_1293:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        memo = {}
        m, n = len(grid), len(grid[0])
        visit = [[0] * n for _ in range(m)]

        def foo(i, j, k, visit):
            nonlocal memo
            print(i, j, k)
            # visit[i][j][k] = 1
            if (i, j, k) in memo:
                return memo[i, j, k]
            if i == m - 1 and j == n - 1:
                return 0
            if k < 0:
                return float('inf')
            r = float('inf')
            for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                if 0 <= x < m and 0 <= y < n:
                    if grid[x][y] == 1 and k > 0 and visit[x][y] == 0:
                        visit[x][y] = 1
                        r = min(r, 1 + foo(x, y, k - 1, visit))
                        visit[x][y] = 0
                    elif grid[x][y] == 0 and visit[x][y] == 0:
                        visit[x][y] = 1
                        r = min(r, 1 + foo(x, y, k, visit))
                        visit[x][y] = 0
            memo[i, j, k] = r
            return r

        res = foo(0, 0, k, visit)
        return res if res < float('inf') else -1

    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        # 使用 BFS
        m, n = len(grid), len(grid[0])
        que = collections.deque([[0, 0, k]])
        visit = set()
        lv = -1
        while que:
            L = len(que)
            lv += 1
            for _ in range(L):
                i, j, r = que.popleft()
                if i == m - 1 and j == n - 1:
                    return lv
                for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                    if 0 <= x < m and 0 <= y < n:
                        if grid[x][y] == 1 and r > 0 and (x, y, r - 1) not in visit:
                            que.append([x, y, r - 1])
                            visit.add((x, y, r - 1))
                        elif grid[x][y] == 0 and (x, y, r) not in visit:
                            que.append([x, y, r])
                            visit.add((x, y, r))
        return -1


# Definition for din QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution_427:
    def construct(self, grid: List[List[int]]) -> 'Node':
        n = len(grid)

        def create(grid, a, b, c, d):
            print(a, b, c, d)
            if a > c or b > d:
                return None
            if a == c and b == d:
                return Node(bool(grid[a][b]), True, None, None, None, None)
            mx, my = (a + c) // 2, (b + d) // 2
            node = Node(False, False, None, None, None, None)
            node.topLeft = create(grid, a, b, mx, my)
            node.topRight = create(grid, a, my + 1, mx, d)
            node.bottomLeft = create(grid, mx + 1, b, c, my)
            node.bottomRight = create(grid, mx + 1, my + 1, c, d)

            nexts = [node.topLeft, node.topRight, node.bottomLeft, node.bottomRight]

            if all(n.isLeaf for n in nexts):
                if all(n.val for n in nexts) or all(n.val == False for n in nexts):
                    node.val = node.topLeft.val
                    node.isLeaf = True
                    node.topLeft = None
                    node.topRight = None
                    node.bottomLeft = None
                    node.bottomRight = None
            return node

        return create(grid, 0, 0, n - 1, n - 1)


class Solution_1366:
    def rankTeams(self, votes: List[str]) -> str:
        n = len(votes[0])
        cnt = {k: [0] * n + [-ord(k)] for k in votes[0]}
        for vote in votes:
            for i, c in enumerate(vote):
                cnt[c][i] += 1

        res = sorted(cnt.values(), reverse=True)
        # print(res)
        return ''.join(chr(-k[n]) for k in res)


class Solution_864:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        # keys => use bitmasking
        # BFS 时将 (i, j, 已经获得的 key) 放入 visit
        m, n = len(grid), len(grid[0])
        start = (0, 0)
        key = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '@':
                    start = (i, j)
                elif ord('a') <= ord(grid[i][j]) <= ord('f'):
                    key += 1

        que = {(start[0], start[1], 0)}
        visit = {(start[0], start[1], 0)}
        step = 0

        while que:
            # print(que)
            next_lv = set()
            for i, j, c in que:
                if c == 2 ** key - 1:
                    return step

                for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] != '#':
                        ch = grid[x][y]
                        new_c = c
                        if ord('A') <= ord(ch) <= ord('F') and not (new_c & (1 << (ord(ch.lower()) - ord('a')))):
                            continue
                        if ord('a') <= ord(ch) <= ord('f'):
                            new_c |= (1 << (ord(ch) - ord('a')))
                        if (x, y, new_c) not in visit:
                            visit.add((x, y, new_c))
                            next_lv.add((x, y, new_c))
            step += 1
            que = next_lv
        return -1


class Solution_1333:
    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> \
            List[int]:
        res = [x for x in restaurants if x[2] >= veganFriendly and x[3] <= maxPrice and x[4] <= maxDistance]

        res = sorted(res, key=lambda x: (-x[1], -x[0]))
        return [x[0] for x in res]


class Solution_488:
    # 下面的代码是错误的但依然能够 AC, 对于一个Case:
    #    board = "RRWWRRBBRR"; hand = "WB"
    # 按照代码的思路是这样子：
    #    RRWW[W]RRBBRR ->  RRRRBB[B]RR -> RRRRRR -> empty
    # 或 RRWWRRBB[B]RR ->  RRWW[W]RRRR -> RRRRRR -> empty
    # 而实际上，在第一次消除时，两个连续RR.RR 拼接到了一起，超过 3 个就会消除，所以这样会剩下两个 RR
    # 正确的思路是 RRWWRRBBR[W]R -> RRWWRRBB[B]R[W]R -> RRWWRRR[W]R -> RRWW[W]R -> RRR -> empty
    # 这样最终的答案都是插入两次，因此下面的代码也能够 AC

    # 考虑另外一个 Case:
    #    board = "WWRRWWRRWW", hand = "RRBBB"
    # 使用代码的思路插入是 2 个，但是并不能正确，正确的思路是：
    # WWRRW[B]WRRWW -> WWR[R]RW[B]WR[R]RWW -> WWW[B]WWW -> [B] -> [B][B][B] -> empty
    # 这样插入的次数是 5.
    def findMinStep(self, board: str, hand: str) -> int:
        # ball, R, Y, B, G, W
        # 0 < len(board) <= 16
        # 0 < len(hand) <= 5
        # 找出插入最少的球将 board 消除完，否则返回 -1
        res = len(hand) + 1

        def dfs(board, hand, cnt):
            nonlocal res
            if not board:
                res = min(res, cnt)
            last = None
            for i, c in enumerate(board):
                if last == c:
                    j = i + 1
                    while j < len(board) and board[j] == c:
                        j += 1
                    if j - (i - 1) >= 3:
                        dfs(board[:i - 1] + board[j:], hand, cnt)
                    elif j - (i - 1) >= 2 and c in hand:
                        hand.remove(c)
                        dfs(board[:i - 1] + board[j:], hand, cnt + 1)
                        hand.append(c)
                else:
                    if hand.count(c) >= 2:
                        hand.remove(c)
                        hand.remove(c)
                        dfs(board[:i] + board[i + 1:], hand, cnt + 2)
                        hand.append(c)
                        hand.append(c)
                last = c

        dfs(board, list(hand), 0)
        return res if res <= len(hand) else -1


class Solution_491:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = set()

        def dfs(idx, tmp):
            nonlocal res
            if idx >= len(nums):
                if len(tmp) >= 2:
                    res.add(tuple(tmp))
                return

            if not tmp or tmp[-1] <= nums[idx]:
                dfs(idx + 1, tmp + [nums[idx]])

            dfs(idx + 1, tmp)

        dfs(0, [])
        return list(res)


class Solution_1090:
    def largestValsFromLabels(self, values: List[int], labels: List[int], num_wanted: int, use_limit: int) -> int:
        # |S| <= num_wanted
        A = sorted([(v, L) for v, L in zip(values, labels)], key=lambda x: -x[0])
        cnt = collections.defaultdict(int)
        i = res = 0
        while num_wanted and i < len(A):
            if cnt[A[i][1]] + 1 <= use_limit:
                num_wanted -= 1
                cnt[A[i][1]] += 1
                res += A[i][0]
            i += 1
        return res


class Solution_886:
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        # 二部图, 使用 染色法
        G = collections.defaultdict(list)
        for a, b in dislikes:
            G[a].append(b)
            G[b].append(a)

        color = {}  # 存储染色的结果

        def dfs(node, c=0):
            if node in color:
                return color[node] == c

            color[node] = c
            return all(dfs(v, c ^ 1) for v in G[node])

        return all(dfs(n) for n in range(1, N + 1) if n not in color)


######################################################################################################################
# heap

class Solution_378:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        # 简单做法，转化层一维数组排序处理
        s = sorted(sum(matrix, []))
        # print(s)
        return s[k - 1]

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        heap = []
        # 对第一行建堆，然后使用每一列的下一个替换堆中的元素，类似于merge k sorted list。
        for j in range(n):
            heapq.heappush(heap, (matrix[0][j], 0, j))

        # 弹出堆顶，替换为所在列的下一个
        print(heap)
        for i in range(k - 1):
            t = heapq.heappop(heap)
            print(t)
            if t[1] != n - 1:
                heapq.heappush(heap, (matrix[t[1] + 1][t[2]], t[1] + 1, t[2]))
        # print(heap)
        return heapq.heappop(heap)[0]


class Solution_373:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # 这是个跟上面一样的题目，至少解法是一样的
        # 使用一个堆先处理完一个 arr， 然后处理第二个 arr，依然是 merge K sorted list
        i = j = 0
        if not nums1 or not nums2 or k <= 0:
            return []
        res = []
        heap = []
        for j in range(len(nums2)):
            heapq.heappush(heap, (nums1[0] + nums2[j], 0, j))
        # print(heap)
        for i in range(min(k, len(nums1) * len(nums2))):
            s, x, y = heapq.heappop(heap)
            res.append([nums1[x], nums2[y]])
            if x != len(nums1) - 1:
                heapq.heappush(heap, (nums1[x + 1] + nums2[y], x + 1, y))

        return res


class Node:
    def __init__(self, val, p):
        self.val = val
        self.p = p

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val

    def __repr__(self):
        return f'{self.val}'


class Solution_23:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None

        dummy = ListNode(None)
        # 直接的 listNode 不能支持排序，然后新建的类 Node ，新建的类支持排序
        heap = []
        for n in lists:
            if n:
                heapq.heappush(heap, Node(n.val, n))
        # print(heap)
        p = dummy
        while heap:
            # print(heap)
            node = heapq.heappop(heap)
            p.next = ListNode(node.val)
            p = p.next

            if node.p.next:
                heapq.heappush(heap, Node(node.p.next.val, node.p.next))

        return dummy.next


class Solution_264:
    def nthUglyNumber(self, n: int) -> int:
        # 这个和 merge K sorted list 一样的
        heap = [1]
        i = 0
        while i < n:
            i += 1
            # print(heap)
            t = heapq.heappop(heap)
            while heap and heap[0] == t:
                heapq.heappop(heap)
            for x in [2, 3, 5]:
                heapq.heappush(heap, t * x)

        return t


class Solution_632:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        # 依然差不多是 merge k sorted list 的思路
        pool = []
        m = float('-inf')
        for i, ls in enumerate(nums):
            if ls:
                m = max(m, ls[0])
                heapq.heappush(pool, (ls[0], i, 0))

        minx, miny = 0, float('inf')
        while pool:
            t, i, j = heapq.heappop(pool)
            # 适用堆保证了 a < c && b - a == d - c，因为小的先入堆
            if miny - minx > m - t:  # 如果区间长度更小的话，更新区间
                minx, miny = t, m

            if j + 1 >= len(nums[i]):
                break
            heapq.heappush(pool, (nums[i][j + 1], i, j + 1))
            m = max(m, nums[i][j + 1])

        return [miny, minx]


class Solution_857:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], K: int) -> float:
        # 需要按照 quality 付工资，每个人至少拿到最低期望薪资
        # 使用 heap 处理的大多是对有序序列的问题？
        from fractions import Fraction
        # 按照比例 升序 排序
        workers = sorted((Fraction(w, q), q, w) for q, w in zip(quality, wage))

        ans = float('inf')
        pool = []
        sumq = 0
        for ratio, q, w in workers:
            heapq.heappush(pool, -q)  # 取负号，大顶堆
            sumq += q  # 当前数量的和

            # 队列中数据个数大于 K 个， pop， 保持 K 个，就是所选的
            if len(pool) > K:
                sumq += heapq.heappop(pool)  # 这里加上弹出的 -q, 其实是减去

            # 队列中数据等于 K , 并且当前一个 ration, q, w -> q 已经在队列中
            # 所以取一下当前最小值
            if len(pool) == K:
                ans = min(ans, ratio * sumq)

        return float(ans)


class Solution_719:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        # [1, 1, 3] -> [0, 2, 2]
        def possible(guess):
            # Is there k or more pairs with distance <= guess
            count = left = 0
            # 滑动窗口
            for right, x in enumerate(nums):
                while x - nums[left] > guess:
                    left += 1
                count += right - left
            return count >= k

        nums.sort()
        lo = 0
        hi = nums[-1] - nums[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if possible(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


class Solution_778:
    def swimInWater(self, grid: List[List[int]]) -> int:
        # 找一条从左上到右下的路径，使得路径上的最大值最小
        # DFS, TLE
        res = float('inf')

        def dfs(i, j, path, max_cur, visit):
            nonlocal res
            if i == j == n - 1:
                # print(path)
                res = min(res, max(path))
                return
            if max_cur > res:
                return
            visit[i][j] = 1
            for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                if 0 <= x < n and 0 <= y < n and visit[x][y] == 0:
                    dfs(x, y, path + [grid[x][y]], max(max_cur, grid[x][y]), visit)

            visit[i][j] = 0

        n = len(grid)
        dfs(0, 0, [grid[0][0]], 0, [[0] * n for _ in range(n)])
        return res

    def swimInWater(self, grid: List[List[int]]) -> int:
        # 使用优先队列求最短路
        N = len(grid)
        pool = [(grid[0][0], 0, 0)]
        seen = set((0, 0))
        res = 0
        while pool:
            cur, i, j = heapq.heappop(pool)
            res = max(res, cur)
            if i == j == N - 1:
                return res
            for x, y in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
                if 0 <= x < N and 0 <= y < N and (x, y) not in seen:
                    seen.add((x, y))
                    heapq.heappush(pool, (grid[x][y], x, y))
        return res


class Solution_787:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        graph = collections.defaultdict(list)
        for u, v, w in flights:
            graph[u].append((w, v))

        pool = [(0, src, 0)]  # cost, node, stops
        res = float('inf')
        while pool:
            cost, node, stop = heapq.heappop(pool)
            if node == dst:
                res = min(res, cost)
            if stop <= K and cost < res:
                for w, v in graph[node]:
                    heapq.heappush(pool, (cost + w, v, stop + 1))

        return res if res < float('inf') else -1


class Solution_692:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        counter = collections.Counter(words)
        pool = [(-freq, word) for word, freq in counter.items()]
        heapq.heapify(pool)
        res = []
        while len(res) < k:
            _, w = heapq.heappop(pool)
            res.append(w)
        return res


class Solution_493:
    # 二分查找
    def reversePairs(self, nums: List[int]) -> int:
        if not nums:
            return 0
        pool = [nums[0]]
        res = 0
        # idx = bisect.bisect_right(arr, x)
        #  x < arr[idx] and x >= arr[idx-1] if x > 0
        for j in range(1, len(nums)):
            i = bisect.bisect_right(pool, 2 * nums[j])
            if i < len(pool):
                res += len(pool) - i
            bisect.insort(pool, nums[j])
        return res


class Solution_315:
    # 二分查找
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if n <= 1:
            return [0] * n
        counts = [0] * n
        pool = [nums[-1]]
        for i in range(n - 2, -1, -1):
            counts[i] = bisect.bisect_left(pool, nums[i])
            bisect.insort(pool, nums[i])
        return counts


class Solution_327:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        # O(n^2) TLE
        cnt = 0
        for i in range(len(nums)):
            tmp = 0
            for j in range(i, len(nums)):
                tmp += nums[j]
                if lower <= tmp <= upper:
                    cnt += 1
        return cnt

    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        # divide & conquer
        def merge(A, lo, mid, hi):
            tmp = A[lo:hi + 1]
            i, j = lo, mid + 1
            for k in range(lo, hi + 1):
                if j > hi or (i <= mid and tmp[i - lo] < tmp[j - lo]):
                    A[k] = tmp[i - lo]
                    i += 1
                else:
                    A[k] = tmp[j - lo]
                    j += 1

        def merge_sort(pre, lo, hi):
            if lo >= hi: return 0
            mid = lo + (hi - lo) // 2
            count = merge_sort(pre, lo, mid) + merge_sort(pre, mid + 1, hi)
            m = n = mid + 1

            for i in range(lo, mid + 1):
                while m <= hi and pre[m] - pre[i] < lower:
                    m += 1
                while n <= hi and pre[n] - pre[i] <= upper:
                    n += 1
                count += n - m  # i + 到 [m, n) 范围内的都是在 lower~upper 内的
            merge(pre, lo, mid, hi)
            return count

        pre = [0]
        for x in nums:
            pre.append(pre[-1] + x)
        return merge_sort(pre, 0, len(nums))


class Solution_354:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # [[2,3], [5, 4], [6,4],[6,5],[7,7]]
        # 这个题目是 LIS 的二维版本
        # DP, O(n^2) 超时
        if not envelopes:
            return 0
        en = sorted(envelopes)
        n = len(en)
        dp = [1] * n
        m = 1
        for i in range(len(en)):
            for j in range(i):
                if en[j][0] < en[i][0] and en[j][1] < en[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # should do O(n log n) at least
        # 思路就是反过来，对部分答案进行二分查找
        # 然后排序的时候对 e[0] 升序，遇到 e[0] 相同的按照 e[1] 降序排列，
        # 这样我们保证顺序处理时，e[0] 总是递增的，因此只需要单独查找 e[1]
        # 对每个 e[1], 我们在当前答案子集中搜索插入（替换）位置，如果比当前最
        # 大的结果还大，那么就加入结果集合中（append）
        if not envelopes: return 0
        en = sorted(envelopes, key=lambda x: (x[0], -x[1]))
        pool = []
        for i, e in enumerate(en):
            idx = bisect.bisect_left(pool, e[1])
            if idx < len(pool):
                pool[idx] = e[1]
            else:
                pool.append(e[1])
        return len(pool)


class Solution_871:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        # DP
        # 我们用 dp(i) 表示使用 i 个加油站能够达到的最远距离， 注意这个题目中车子携带的
        # 油箱是无穷大的。
        # 当我们不使用加油站，dp(0) = startFuel # 初始值
        #
        dp = [startFuel] + [0] * len(stations)
        for i, (location, capacity) in enumerate(stations):
            for t in range(i, -1, -1):
                if dp[t] >= location:
                    dp[t + 1] = max(dp[t + 1], dp[t] + capacity)

        for i, d in enumerate(dp):
            if d >= target:
                return i
        return -1

    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        # 用 heap 来解决，记录每个加油站中油的升数，保存在一个 max_heap 中
        pool = []
        stations.append((target, float('inf')))  # 将 target 节点当作有无穷的多的燃料

        ans = prev = 0
        tank = startFuel
        for loc, capacity in stations:
            tank -= (loc - prev)  # 先向前走，不加燃料，如果路程中发现没油了，说明前面需要加燃料，
            # 加前面遇到的加油站中油量多的
            while pool and tank < 0:
                tank += - heapq.heappop(pool)
                ans += 1
            # 加完前面所有的也不够
            if tank < 0:  # //不能抵达当前位置
                return -1
            # 把当前遇到的一个放到堆里
            heapq.heappush(pool, -capacity)
            prev = loc
        return ans


import random


class Solution_710:

    def __init__(self, N: int, blacklist: List[int]):
        # do a remap, map[blacklist[i]] -> (N-len(blacklist), N) 且不在 blacklist 中的数
        self.memo = {x: -1 for x in blacklist}
        m = N - len(blacklist)
        for b in blacklist:
            if b < m:  # 只有这些需要做 remap
                while (N - 1) in self.memo:
                    N -= 1
                self.memo[b] = N - 1
                N -= 1
        self.m = m

    def pick(self) -> int:
        res = random.randint(0, self.m - 1)
        while res in self.memo:
            return self.memo[res]
        return res


class Solution_483:
    def smallestGoodBase(self, n: str) -> str:
        """
            $n = b^m + b^(m-1) + ... + b + 1$
            $n - 1 = b(b^(m-1) + b^(m-2) + ... + b + 1)$
            $n - b^m = b^(m-1) + b^(m-2) + ... + b + 1$
            $n - 1 = b(n-b^m)$
            $n = (b^(m+1) - 1)/(b-1)$
        And:
            $b^m < n = b^m + b^(m-1) + ... + b + 1 < (b+1)^m$
            $b^m < n < (b+1)^m$
            $b < n^(1/m) < b+1$
            因此为了找到合适的 b, 我们只需要检查 int(n^(1/m)) 是否满足条件（向下取整有可能等于 b）
        Other:
            最小的 base 是 2, 因此 m 一定在 [2, int(log(n, 2))] 之间，而且根据 m 和候选 b之间的关系
            可以知道，m 越大，b 越小，因此从大的 m 往小的搜索，遇到满足条件就返回。而最终没有找到的话
            就返回 b = n-1.
        """
        n = int(n)
        max_m = int(math.log(n, 2))
        for m in range(max_m, 1, -1):
            base = int(n ** (1 / m))

            if (base ** (m + 1) - 1) // (base - 1) == n:
                return str(base)

        return str(n - 1)


class Solution_525:
    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [-2] * (2 * n + 1)  # -n ~ +n 所以是 2 倍的 n
        dp[n] = -1
        m = cnt = 0
        for i in range(n):
            cnt += 1 if nums[i] else -1
            if dp[cnt + n] >= -1:
                m = max(m, i - dp[cnt + n])
            else:
                dp[cnt + n] = i
        return m

    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        m = cnt = 0
        memo = {}
        for i in range(n):
            cnt += 1 if nums[i] else -1
            if cnt == 0:
                m = i + 1  # 如果 count 是 0，从头开始到现在就是最长的
            if cnt in memo:
                m = max(m, i - memo[cnt])
            else:
                memo[cnt] = i
        return m


class Solution_554:
    def leastBricks(self, wall: List[List[int]]) -> int:

        pool = []
        for i, w in enumerate(wall):
            heapq.heappush(pool, (w[0], i, 0))
        res = len(wall)
        while pool:
            w, i, j = heapq.heappop(pool)
            if w == sum(wall[0]):
                break
            tmp = 0
            for _w, _i, _j in pool:
                if _w > w:
                    tmp += 1
            # print(tmp)
            res = min(res, tmp)
            while pool and w == pool[0][0]:
                t, x, y = heapq.heappop(pool)
                if y + 1 >= len(wall[x]):
                    continue
                # print(x, y+1)
                heapq.heappush(pool, (w + wall[x][y + 1], x, y + 1))

            if j + 1 >= len(wall[i]):
                continue
            heapq.heappush(pool, (w + wall[i][j + 1], i, j + 1))
        return res

    def leastBricks(self, wall: List[List[int]]) -> int:
        # 用一个hash map 记录到共同位置的有多少个
        # 比如 例子 位置 1 有三个，穿过的就有 len(wall) - 3 个

        if not wall: return 0
        count = 0
        memo = collections.defaultdict(int)
        for row in wall:
            tmp = 0
            for i in range(len(row) - 1):
                tmp += row[i]
                memo[tmp] += 1
                count = max(count, memo[tmp])
        return len(wall) - count


class Solution_423:
    def originalDigits(self, s: str) -> str:
        # 0, 2, 4, 6, 8 都有独特的字母，z, w, u, x, g
        # 我们删除了上面这些数字对应的字母之后，剩下的奇数中
        # o->1, t/h/r->3, f->5, 7->s, 剩下的就是 9
        count = [0] * 10
        memo = {'z': (0, 'zero'), 'w': (2, 'two'), 'u': (4, 'four'), 'x': (6, 'six'), 'g': (8, 'eight'),
                'o': (1, 'one'), 't': (3, 'three'), 'f': (5, 'five'), 's': (7, 'seven')}
        cnt = collections.Counter(s)

        for ls in [['z', 'w', 'u', 'x', 'g'], ['o', 't', 'f', 's']]:
            for c in ls:
                num, t = memo[c]
                if c in cnt:
                    count[num] = cnt[c]
                    for ch in t:
                        cnt[ch] -= count[num]
                        if cnt[ch] == 0:
                            cnt.pop(ch)

        if 'i' in cnt:
            count[9] = cnt['i']

        return ''.join(str(i) * count[i] for i in range(10))


class Solution_421:
    def findMaximumXOR(self, nums: List[int]) -> int:
        # O(n^2), TLE
        m = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                m = max(m, nums[i] ^ nums[j])
        return m

    def findMaximumXOR(self, nums: List[int]) -> int:
        # O(n) ，对每一个数操作 bit 是 O(32*n)
        if not nums: return 0
        # 前缀树，每个节点可能有 0/1 两个孩子，深度为 31
        # 如果完全扩展的话，这个树节点数巨大，但是数相对比较少的情况下，并不会完全扩展。
        trie = {}
        m = 0
        for x in nums:
            complement = cur = trie
            val = 0
            # 从最高位到最低位开始处理
            for i in range(31, -1, -1):
                bit = (x >> i) & 1  # 当前位的取值
                if bit not in cur:  # 新建分支
                    cur[bit] = {}
                cur = cur[bit]

                # 对应补位存在（另一个分支，对应数组中的另一个数）
                if (1 - bit) in complement:
                    complement = complement[1 - bit]
                    val += (1 << i)  # 结果里可以加上这一位
                else:
                    complement = complement[bit]  # 不存在的话同时向下同时这一位是 0
            m = max(m, val)
        return m

    def findMaximumXOR(self, nums: List[int]) -> int:
        # https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/discuss/91050/Python-6-lines-bit-by-bit
        # 基本想法是从最高位到最低位一位一位建立起最终答案
        # (ans^1^p) in prefix 表示有一个 q in prefix 使得 (ans^1^p) = q  => ans^1^p^q = p^q = ans ^ 1
        # 即表示在前缀集合中有 p^q = ans ^ 1, 而 ans 刚刚前移一位 最低位一定 0， ans^1 就是 ans 讲最低位置1
        # 所以这一个语句等价于：
        # for p in prefix:
        #     for q in prefix:
        #         if p^q == ans^1:
        #             ans += 1
        #             break
        ans = 0
        for i in range(31, -1, -1):
            ans <<= 1
            # 前缀
            prefix = {x >> i for x in nums}
            #
            ans += any((ans ^ 1 ^ p) in prefix for p in prefix)
        return ans


class Solution_1365:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        arr = sorted([(x, i) for i, x in enumerate(nums)])
        res = [0] * len(arr)
        last, last_idx = float('-inf'), -1
        for i, (x, idx) in enumerate(arr):
            # print(i, x, idx, i)
            if x == last:
                res[idx] = res[last_idx]
            else:
                res[idx] = i
            last, last_idx = x, idx
        return res


class Solution_318:
    def maxProduct(self, words: List[str]) -> int:
        # 只有 lowercase, 用 26 位表示
        bits = []
        for w in words:
            tmp = 0
            for c in w:
                tmp |= (1 << (ord(c) - ord('a')))
            bits.append(tmp)

        res = 0
        for i, wa in enumerate(words):
            for j, wb in enumerate(words):
                if bits[i] & bits[j] == 0 and len(wa) * len(wb) > res:
                    res = len(wa) * len(wb)
        return res


class Solution_547:
    def findCircleNum(self, M: List[List[int]]) -> int:

        if not M: return 0
        m, n = len(M), len(M[0])

        visit = [0] * n

        def dfs(v):
            visit[v] = 1
            for node in G[v]:
                if visit[node] == 0:
                    dfs(node)

        res = 0
        # 建立邻接表
        G = collections.defaultdict(list)
        for i in range(m):
            for j in range(n):
                if i != j and M[i][j] == 1:
                    G[i].append(j)
                    G[j].append(i)
        # DFS 找连通分量数
        for i in range(n):
            if visit[i] == 0:
                dfs(i)
                res += 1
        return res


class Solution_540:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        # O(n)
        from functools import reduce
        return reduce(lambda x, y: x ^ y, nums)

    def singleNonDuplicate(self, nums: List[int]) -> int:
        # 数组元素是 2k+1 个，所以一定是奇数个，
        # 我们检查偶数位置和后一个位置
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if mid & 1:
                mid -= 1

            if nums[mid] != nums[mid + 1]:
                # 如果不相等，mid 或者左半部分是答案
                hi = mid
            else:
                # 如果相等的话，前面的都是相等对，所以答案在后面
                lo = mid + 2

        return nums[lo]


class Solution_1311:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> \
            List[str]:
        # BFS first, 然后搜集层的结果
        n = len(friends)
        visit = [0] * n

        lv = 0
        que = collections.deque()
        que.append(id)
        visit[id] = 1
        while que:
            L = len(que)
            if lv == level:
                break
            for _ in range(L):
                t = que.popleft()
                for v in friends[t]:
                    if visit[v] == 0:
                        visit[v] = 1
                        que.append(v)
            lv += 1
        memo = collections.defaultdict(int)
        for t in que:
            for video in watchedVideos[t]:
                memo[video] += 1
        # 排序按照 freq 排，相同 freq 的按照字典序
        return sorted(memo.keys(), key=lambda x: (memo[x], x))


class Solution:
    # follow up 就是蓄水池抽样，从流式数据中抽取 k 个样本，被抽中的概率是 k/n
    # https://leetcode.com/problems/linked-list-random-node/discuss/85659/Brief-explanation-for-Reservoir-Sampling
    # 步骤如下:
    #   1. 首先选取 1,2,...,k 放入样本池子
    #   2. 对于第 k+1 个样本，以 k/(k+1) 的概率选它，然后以 1/k 的概率替换掉池子中的一个
    #   3. 对于第 k+i 个样本，（k+i >= k），以 k/(k+i) 的概率选他，然后以 1/k 的概率替换掉池子中的一个。
    #
    # 证明：
    #   最终一个数 X 最后被选中的概率是：前一次被选中 x 后一次不被替换
    #       k/(k+i-1) x (1 - k/(k+i) x 1/k)
    #       = k/(k+i)
    #   当 k+i 等于 n 的时候，最终的概率是 k/n
    # 这个题是 k = 1 的特例
    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        # self.ls = []
        # while head:
        #     self.ls.append(head.val)
        #     head = head.next
        self.head = head

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        # return random.choice(self.ls)
        res = self.head.val  # 池子
        p = self.head.next
        k = 1
        i = 1
        while p:
            x = random.random()
            y = k / (k + i * 1.0)
            if x <= y:
                res = p.val
            i += 1
            p = p.next
        return res


class Solution_609:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        # n >= 1 一个目录下至少一个文件
        # m >= 0，m==0 时就只有一个root 路径

        # map file_content to filename
        memo = collections.defaultdict(list)
        for raw in paths:
            ls = raw.split()
            d = ls[0]
            for t in ls[1:]:
                idx = t.index('(')
                content = t[idx + 1:-1]
                filename = t[:idx]
                memo[content].append(d + '/' + filename)

        return [v for v in memo.values() if len(v) > 1]


class Solution_1300:

    def findBestValue(self, arr: List[int], target: int) -> int:
        n = len(arr)
        arr = sorted(arr, reverse=True)
        m = arr[0]  # len(A) >= 1
        while arr and target >= arr[-1] * len(arr):
            target -= arr.pop()  # 排除掉最小值，因为不会影响了，
        # 大于 target 之后，就讲剩余的 target 排除掉
        # 如果 A 为空的话，答案就是 m, 否则的话就是 target / len(arr) 最近的值
        # 可以使用 round
        return round((target - 1e-5) / len(arr)) if arr else m

    def findBestValue(self, arr: List[int], target: int) -> int:
        n = len(arr)
        arr.sort()
        i = 0
        while i < n and target > arr[i] * (n - i):
            target -= arr[i]
            i += 1

        if i == n:
            return arr[n - 1]

        tmp = target // (n - i)
        if target - tmp * (n - i) > (tmp + 1) * (n - i) - target:
            return tmp + 1
        return tmp


# Your Solution object will be instantiated and called as such:
# obj = Solution(N, blacklist)
# param_1 = obj.pick()

class Solution_502:
    def findMaximizedCapital(self, k: int, W: int, Profits: List[int], Capital: List[int]) -> int:
        # 每次选 当前能启动盈利最大的项目，选 k 次
        # 暴力方法, TLE
        s = [(c, p) for c, p in zip(Capital, Profits)]
        for _ in range(k):
            idx = -1
            max_p = 0
            for i, (c, p) in enumerate(s):
                if c <= W and p > max_p:
                    max_p = p
                    idx = i
            W += max_p
            if 0 <= idx < len(s):
                s.pop(idx)
            else:
                break
        return W

    def findMaximizedCapital(self, k: int, W: int, Profits: List[int], Capital: List[int]) -> int:
        pc = sorted(zip(Profits, Capital), key=lambda x: x[1])
        pool = []
        i = 0
        for _ in range(k):
            while i < len(pc) and pc[i][1] <= W:
                heapq.heappush(pool, -pc[i][0])
                i += 1
            if pool:
                W -= heapq.heappop(pool)
        return W


class Solution_767:
    def reorganizeString(self, S: str) -> str:
        # 和 1054 是同一道题
        counter = collections.Counter(S)
        if max(counter.values()) > len(S) - max(counter.values()) + 1:
            return ''

        res = [''] * len(S)

        pool = [(-v, k) for k, v in counter.items()]
        heapq.heapify(pool)

        v, k = heapq.heappop(pool)
        v = -v
        index = 0
        while index < len(S):
            if v <= 0:
                if not pool:
                    break
                v, k = heapq.heappop(pool)
                v = -v
            v -= 1
            res[index] = k
            index += 2
            if index >= len(S):
                index = 1
        return ''.join(res)


class Solution_1054:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        counter = collections.Counter(barcodes)
        pool = [(-v, k) for k, v in counter.items()]
        heapq.heapify(pool)
        # print(pool)
        res = [0] * len(barcodes)
        v, k = heapq.heappop(pool)
        v = -v

        index = 0
        while index < len(barcodes):
            if v <= 0:
                if not pool:
                    break
                v, k = heapq.heappop(pool)
                v = -v

            v -= 1

            res[index] = k
            # print(res)
            index += 2
            if index >= len(barcodes):
                index = 1

        return res


class Solution_743:
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        # 建立图， BFS
        graph = collections.defaultdict(list)
        for u, v, w in times:
            graph[u].append((w, v))

        dist = {x: float('inf') for x in range(1, N + 1)}

        def dfs(node, t):
            if t >= dist[node]: return
            dist[node] = t
            for w, k in sorted(graph[node]):
                dfs(k, t + w)

        dfs(K, 0)
        ans = max(dist.values())
        return ans if ans < float('inf') else -1

    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        graph = collections.defaultdict(list)
        for u, v, w in times:
            graph[u].append((w, v))

        dist = {x: float('inf') for x in range(1, N + 1)}

        que = collections.deque([[K, 0]])
        while que:
            v, t = que.popleft()
            if t < dist[v]:
                dist[v] = t
                for w, k in graph[v]:
                    que.append([k, t + w])
        ans = max(dist.values())
        return ans if ans < float('inf') else -1


class Solution_1094:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        trips.sort(key=lambda x: x[1])
        pool = []
        for num, sloc, eloc in trips:
            while pool and pool[0][0] <= sloc:
                e, x = heapq.heappop(pool)
                capacity += x
            # print(capacity)
            if capacity - num < 0:
                return False
            capacity -= num
            heapq.heappush(pool, (eloc, num))
        return True


class Solution_1368:
    def minCost(self, grid: List[List[int]]) -> int:
        # m, n \in [1, 100]
        # BFS + DFS
        # 先用 DFS 找出所有能够达到点，将这些点放入数组 bfs
        # 然后遍历 bfs 数组修改并继续 DFS
        m, n = len(grid), len(grid[0])
        inf = 10 ** 9
        dp = [[inf] * n for _ in range(m)]
        k = 0
        #              Right    Left    Down     Up
        dirt = [None, (0, 1), (0, -1), (1, 0), (-1, 0)]
        bfs = []

        def dfs(x, y):
            if 0 <= x < m and 0 <= y < n and dp[x][y] == inf:  # 这里dp 充当了visit 的作用
                dp[x][y] = k
                bfs.append([x, y])
                dfs(x + dirt[grid[x][y]][0], y + dirt[grid[x][y]][1])

        dfs(0, 0)
        while bfs:
            k += 1
            bfs, bfs2 = [], bfs  # 这个其实就是每次出队列这一层
            for x, y in bfs2:
                for i, j in dirt[1:]:
                    dfs(x + i, y + j)
        return dp[-1][-1]


class Solution_1162:
    def maxDistance(self, grid: List[List[int]]) -> int:
        # 从所有 grid[i][j] == 1 的节点同时出发做 BFS
        # 最深的层数就是答案
        N = len(grid)
        inf = 201
        dist = [[inf] * N for _ in range(N)]
        bfs = set()
        for i, j in itertools.product(range(N), range(N)):
            if grid[i][j] == 1:
                bfs.add((i, j))
                dist[i][j] = 0

        if len(bfs) == 0 or len(bfs) == N * N:
            return -1
        k = 0
        while bfs:
            k += 1
            bfs, bfs2 = set(), bfs
            for x, y in bfs2:
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < N and 0 <= j < N and dist[i][j] == inf:
                        dist[i][j] = k
                        bfs.add((i, j))
        return k - 1


class Solution_1036:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        # 直接 BFS 肯定超时，还可能超内存
        # 考虑 block 作为一条线阻隔了 source 和 target, blocked 的长度不超过 200
        # 因此被 blocked 挡住的区域是有限的，最大 19900 个格子（不算blocked本身)
        # 所以我们用 BFS/DFS 搜索，探索超过 20000 > 19900 格子之后，就可以认为 source -> target
        # 是联通的。
        if not blocked:
            return True

        blocked = set(map(tuple, blocked))

        def bfs(src, dst):
            que = collections.deque([src])
            visited = {tuple(src)}
            k = 0
            while que and k < 20000:
                k += 1
                x, y = que.popleft()
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < 10 ** 6 and 0 <= j < 10 ** 6 and (i, j) not in blocked and (i, j) not in visited:
                        if [i, j] == dst:
                            return True
                        visited.add((i, j))
                        que.append([i, j])
            return k >= 20000

        return bfs(source, target) and bfs(target, source)


class Solution_934:
    def shortestBridge(self, A: List[List[int]]) -> int:
        # 只有两个岛，首先 DFS 找出两个岛屿，然后从其中一个 BFS计算距离,
        n = len(A)
        visit = [[0] * n for _ in range(n)]

        def dfs(x, y, island):
            island.append((x, y))
            for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= i < n and 0 <= j < n and visit[i][j] == 0 and A[i][j] == 1:
                    visit[i][j] = 1
                    dfs(i, j, island)

        lands = []
        for i in range(n):
            for j in range(n):
                if A[i][j] == 1 and visit[i][j] == 0:
                    visit[i][j] = 1
                    island = []
                    dfs(i, j, island)
                    lands.append(set(island))
        # print(lands)
        que = lands[0]
        other = lands[1]
        lv = 0
        while que:
            next_lv = set()
            for x, y in que:
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < n and 0 <= j < n:
                        if (i, j) in other:
                            return lv
                        if visit[i][j] == 0 and A[i][j] == 0:
                            visit[i][j] = 1
                            next_lv.add((i, j))
            que = next_lv
            lv += 1
        return lv - 1  # 不会从这里返回


######################################################################################################################
# sweep line 算法
import sortedcontainers


class Solution_218:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        events = []
        for a, b, h in buildings:
            events.append((a, -h))
            events.append((b, h))

        events.sort()
        bst = sortedcontainers.sortedset.SortedList([0])
        res = []
        cur = [0, 0]
        for x, h in events:
            if h < 0:
                bst.add(h)  # 左边入堆
            else:
                bst.remove(-h)  # 右边出堆

            if -bst[0] != cur[1]:
                cur = [x, -bst[0]]
                res.append([x, -bst[0]])

        return res

    def _getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # 题目的关键点是高度发生变化的第一个点，即对应是矩阵左上角的点，所以
        # 这个题目可以用扫描线问题解决。
        # 首先，维护一个高度的最大堆。我们希望按照左边 L 的大小（从小到达）排序
        # 所以队矩阵按照 [L, -H, R] 存放。
        # 另外，需要考虑的是每一段最后一个点的高度是 0，并且按照 R 从小到大排序
        # 所以另外存放，[R, 0, 0] 需要注意的一点是对 (R, 0, 0) 降重。

        points = [(L, -H, R) for L, R, H in buildings] + [(R, 0, 0) for R in set(r for _, r, _ in buildings)]
        points.sort()
        # 堆的初始值设置高度为 float('inf'), 遇到的第一个就左边点就放入结果集
        # res 第一个位置存放一个 [0,0] 作为哨兵。
        heap, res = [(0, float('inf'))], [[0, 0]]
        for l, nh, r in points:
            while heap[0][1] <= l:  # 堆顶对应 x 坐标小于当前左边 l 坐标
                heapq.heappop(heap)

            # 如果高度不为 0， 即是一个完整的矩形
            if nh:
                heapq.heappush(heap, (nh, r))
            # 如果结果中前面一个的高度，如当前堆顶的高度不一样，把当前左边加入结果集合
            if res[-1][1] != -heap[0][0]:
                res += [[l, -heap[0][0]]]
        return res[1:]

    def getSkyline(self, buildings):
        # 二分法，两个建筑物合并，左边建筑物 [lo_l, lo_h, lo_r] 右边建筑物 [hi_l, hi_h, hi_r]
        # 三种情况：
        # 使用 lh, rh 记录前一个左右建筑物的高度，初始值为 0。
        #   lo_l < hi_l    这时候需要往合并结果集合中添加 [lo_l, max(lo_h, rh)]
        #   lo_l > hi_l    这时候需要往合并结果集合中添加 [lo_l, max(hi_h, lh)]
        #   lo_l == hi_l   这时候需要往合并结果集合中添加 [lo_l, max(hi_h, lo_h)]
        def merge(lo, hi):
            lh = rh = i = j = 0
            res = []
            while i < len(lo) and j < len(hi):
                if lo[i][0] < hi[j][0]:
                    cp = [lo[i][0], max(lo[i][1], rh)]
                    lh = lo[i][1]
                    i += 1
                elif lo[i][0] > hi[j][0]:
                    cp = [hi[j][0], max(hi[j][1], lh)]
                    rh = hi[j][1]
                    j += 1
                else:
                    cp = [lo[i][0], max(lo[i][1], hi[j][1])]
                    lh, rh = lo[i][1], hi[j][1]
                    i += 1
                    j += 1
                if not res or res[-1][1] != cp[1]:
                    res.append(cp)
            res += lo[i:] + hi[j:]
            return res

        def skyline(arr):
            # edge case
            if not arr: return []
            if len(arr) == 1: return [[arr[0][0], arr[0][2]], [arr[0][1], 0]]
            # 分治法
            mid = len(arr) // 2
            lo = skyline(arr[:mid])
            hi = skyline(arr[mid:])
            return merge(lo, hi)

        return skyline(buildings)


from functools import reduce


class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        # 容斥原理
        # |A ∪ B ∪ C| = |A| + |B| + |C| - (|A ∩ B| + |A ∩ C| + |B ∩ C|) + |A ∩ B ∩ C|
        # |∪_{i=1}^{n} A_i| = \sum_{S ⊆ [n] 且 S != 空集} (-1) ^{|S| + 1}| ∩_{i \in S} A_i|
        # 枚举所有的子集，时间复杂度是 O(n * 2^n)
        def intersect(a, b):
            # 矩形 din, b 的交， 如果没有交左下的点会比右上高
            return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]

        def area(rec):
            dx = max(0, rec[2] - rec[0])
            dy = max(0, rec[3] - rec[1])
            return dx * dy

        ans = 0
        for size in range(1, len(rectangles) + 1):
            for g in itertools.combinations(rectangles, size):
                ans += (-1) ** (size + 1) * area(reduce(intersect, g))

        return ans % int(1e9 + 7)

    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        # 扫描线算法，按照 y 轴排序，从下向上扫描
        # 可以使用 线段树 来保存 active 节约时间复杂度
        OPEN, CLOSE = 0, 1
        events = []
        for x1, y1, x2, y2 in rectangles:
            events.append((y1, OPEN, x1, x2))
            events.append((y2, CLOSE, x1, x2))
        events.sort()

        def query(active):
            # 计算一组(active中)有序的 interval 集合并的长度，类似于 merge intervals
            ans = 0
            cur = -1
            for x1, x2 in active:
                cur = max(cur, x1)
                ans += max(0, x2 - cur)
                cur = max(cur, x2)
            return ans

        active = []
        cur_y = events[0][0]
        res = 0
        for y, T, x1, x2 in events:
            res += query(active) * (y - cur_y)

            if T == OPEN:
                active.append((x1, x2))
                active.sort()
            else:
                active.remove((x1, x2))
            cur_y = y
        return res % int(1e9 + 7)

    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        # 水平方向扫描，一样的
        Left, Right = 0, 1
        events = []
        for x1, y1, x2, y2 in rectangles:
            events.append([x1, Left, y1, y2])
            events.append([x2, Right, y1, y2])

        events.sort()

        def calcu(active):
            ans = 0
            cur_y = -1
            for y1, y2 in active:
                cur_y = max(cur_y, y1)
                ans += max(0, y2 - cur_y)
                cur_y = max(cur_y, y2)
            return ans

        active_set = []
        cur_x = events[0][0]
        res = 0
        for x, T, y1, y2 in events:
            res += calcu(active_set) * (x - cur_x)

            if T == Left:
                active_set.append((y1, y2))
                active_set.sort()
            else:
                active_set.remove((y1, y2))
            cur_x = x
        return res % int(1e9 + 7)


class Solution_391:
    def isRectangleCover(self, rectangles: List[List[int]]) -> bool:
        # O(n) 的解法
        #   1. 所有矩形的面积和等于 左下右上矩形点构成的面积和
        #   2. 除了最外面的大矩形外，所有的 corner(一个矩形右四个 corner) 都出现 2 次或者四次
        def area(rec):
            return (rec[2] - rec[0]) * (rec[3] - rec[1])

        total = 0
        bl, tr = (float('inf'), float('inf')), (-1, -1)
        memo = collections.defaultdict(int)
        for rect in rectangles:
            total += area(rect)
            bl = (min(bl[0], rect[0]), min(bl[1], rect[1]))
            tr = (max(tr[0], rect[2]), max(tr[1], rect[3]))
            memo[rect[0], rect[1]] += 1
            memo[rect[0], rect[3]] += 1
            memo[rect[2], rect[1]] += 1
            memo[rect[2], rect[3]] += 1

        if total != (tr[0] - bl[0]) * (tr[1] - bl[1]):
            return False
        for point in [bl, tr, (bl[0], tr[1]), (tr[0], bl[1])]:
            if memo[point] != 1:
                return False
            memo.pop(point)

        for k, v in memo.items():
            if v not in {2, 4}:
                return False
        return True

    def isRectangleCover(self, rectangles: List[List[int]]) -> bool:
        # 扫描线算法
        import bisect
        Left, Right = 1, 0  # 相等的情况时，Right 放在前面，先处理
        bl, tr = [float('inf'), float('inf')], [-1, -1]
        total = 0
        events = []
        for x1, y1, x2, y2 in rectangles:
            events.append([x1, Left, y1, y2])
            events.append([x2, Right, y1, y2])
            total += (x2 - x1) * (y2 - y1)
            bl = [min(bl[0], x1), min(bl[1], y1)]
            tr = [max(tr[0], x2), max(tr[1], y2)]

        events.sort(key=lambda x: [x[0], x[1]])

        active = []
        for x, T, y1, y2 in events:
            if T == Right:
                active.remove((y1, y2))
            else:
                # 处理 overlap
                # 这个 lower_bound 插入位置处理
                ix = bisect.bisect_left(active, (y1, y2))
                # y1 <= active[ix][0] 如果  active[ix] < y2 则有交集
                if ix != len(active) and active[ix][0] < y2:
                    print(active, ix, (y1, y2))
                    return False
                if ix > 0 and active[ix - 1][1] > y1:
                    return False
                bisect.insort(active, (y1, y2))

        return total == (tr[0] - bl[0]) * (tr[1] - bl[1])
