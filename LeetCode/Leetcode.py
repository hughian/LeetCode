import math
from typing import List
import itertools
import collections
import bisect


class KthLargest:
    def __init__(self, k: int, nums):
        self.ls = []
        self.k = k
        for x in nums:
            if len(self.ls) < k:
                # insert
                self._insert(x)
            elif x > self.ls[-1]:
                self._insert(x)
                self.ls.pop()

    def _insert(self, val):
        if len(self.ls) == 0:
            self.ls.append(val)
            return
        elif len(self.ls) == 1:
            if self.ls[0] > val:
                self.ls.append(val)
            else:
                self.ls.insert(0, val)
            return

        left, right = 0, len(self.ls) - 1
        while left < right:
            mid = left + (right - left) // 2
            if val >= self.ls[mid]:
                right = mid
            else:
                left = mid + 1
        if left == len(self.ls) - 1:
            if self.ls[-1] > val:
                self.ls.append(val)
                return
        self.ls.insert(left, val)

    def add(self, val: int) -> int:
        if len(self.ls) < self.k:
            # insert
            self._insert(val)
        elif val > self.ls[-1]:
            self._insert(val)
            self.ls.pop()
        return self.ls[-1]


class Solution_23:
    # TODO 败者树
    def mergeKLists(self, lists):
        n = len(lists)
        ls = [n] * (n + 1)

        def adjust(s, buf):
            """
            :param s:  存胜者
            :param buf:
            :return:
            """
            t = (s + n) // 2
            print('#', s, n, t)
            while t > 0:
                if buf[s].val > buf[ls[t]].val:
                    ls[t], s = s, ls[t]
                t = t // 2
                print(ls)
            ls[0] = s

        def build_ls(buf):
            for i in range(n):
                adjust(i, buf)

        buf = lists + [ListNode(-float('inf'))]
        build_ls(buf)
        print(ls)
        print([p.val for p in buf])
        new_head = ListNode(None)
        q = new_head
        while any([b.val != -float('inf') for b in buf[:-1]]):
            p = buf[ls[0]]
            buf[ls[0]] = p.next if p.next else ListNode(-float('inf'))
            adjust(ls[0], buf)
            q.next = p
            q = q.next
        return new_head.next

    def _mergeKLists(self, lists):
        if len(lists) == 0:
            return None
        if len(lists) > 0 and all([p is None for p in lists]):
            return None

        new_head = ListNode(None)
        q = new_head
        while lists:
            ps = [p for p in lists if p is not None]
            p = min(ps, key=lambda p: p.val)
            ps.remove(p)
            t = p.next
            p.next = None
            q.next = p
            q = q.next
            if t is not None:
                ps.append(t)
            lists = ps[:]
        return new_head.next

    @staticmethod
    def run(Lists):
        s = Solution_23()
        res = []
        for L in Lists:
            head = ListNode(None)
            p = head
            for x in L:
                p.next = ListNode(x)
                p = p.next
            res.append(head.next)
        r = s.mergeKLists(res)

        p = r
        while p:
            print(p.val, end=' ')
            p = p.next
        print('')


class Solution_150:
    def evalRPN(self, tokens) -> int:
        stack = []
        for x in tokens:
            if x in ['+', '-', '*', '/']:
                b = stack.pop()
                a = stack.pop()
                if x == '+':
                    stack.append(a + b)
                elif x == '-':
                    stack.append(a - b)
                elif x == '*':
                    stack.append(a * b)
                else:
                    r = abs(a) // abs(b)
                    r = r * -1 if a * b < 0 else 1
                    stack.append(r)
            else:
                stack.append(int(x))
            # print(stack)
        return stack[0]

    @staticmethod
    def run():
        s = Solution_150()
        s.evalRPN(["3", "4", "5", "*", "+", "6", "-"])
        s.evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"])


class Solution_224:
    def calculate(self, s: str) -> int:
        operator = ['#']
        tokens = []
        n = []
        for c in s:
            if c == ' ':
                continue
            elif c in ['(', ')', '+', '-']:
                if n:
                    num = int(''.join(n))
                    n = []
                    tokens.append(num)
                if c == '(':
                    operator.append(c)
                elif c == ')':
                    while len(operator) > 1 and operator[-1] != '(':
                        tokens.append(operator[-1])
                        operator.pop()
                    operator.pop()
                elif c == '+' or c == '-':
                    if operator[-1] != '(':
                        while len(operator) > 1 and operator[-1] != '(':
                            t = operator[-1]
                            tokens.append(t)
                            operator.pop()
                    operator.append(c)
            else:
                n.append(c)
        if n:
            num = int(''.join(n))
            tokens.append(num)
        if len(operator) > 1:
            tokens.append(operator[-1])
            operator.pop()
        print(tokens)
        stack = []
        for x in tokens:
            if x in ['+', '-']:
                b = stack.pop()
                a = stack.pop()
                r = a + b if x == '+' else a - b
                stack.append(r)
            else:
                stack.append(int(x))
        return stack[0]

    @staticmethod
    def run():
        s = Solution_224()
        r = s.calculate("(11+(4+5+2)-3)+(6+8)")
        print(r)


class Solution_54:
    # 解法1
    def spiralOrder(self, matrix):
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return []

        visit = [[False] * n for _ in matrix]
        res = []
        dirt = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

        i = j = idx = 0
        for _ in range(m * n):
            res.append(matrix[i][j])
            visit[i][j] = True
            ni, nj = i + dirt[idx][0], j + dirt[idx][1]
            if 0 <= ni < m and 0 <= nj < n and not visit[ni][nj]:
                i, j = ni, nj
            else:
                idx = (idx + 1) % 4
                i, j = i + dirt[idx][0], j + dirt[idx][1]
        return res

    def spiralOrder(self, matrix):
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return []

        res = []
        rowu, rowd = 0, m - 1
        coll, colr = 0, n - 1
        while rowu <= rowd and coll <= colr:

            for j in range(coll, colr + 1):  # [rowu][coll, colr]
                res.append(matrix[rowu][j])

            for i in range(rowu + 1, rowd + 1):  # [rowu+1, rowd][colr]
                res.append(matrix[i][colr])

            if rowu < rowd and coll < colr:
                for j in range(colr - 1, coll, -1):  # [rowd][colr-1, colr+1]
                    res.append(matrix[rowd][j])

                for i in range(rowd, rowu, -1):  # [rowd, rowu+1][coll]
                    res.append(matrix[i][coll])

            rowu += 1
            rowd -= 1
            coll += 1
            colr -= 1
        return res


class Solution40:
    # ToDO link to 218 the skyline problem
    def rectangleArea(self, rectangles: "List[List[int]]") -> int:
        # | A ∪ B ∪ C | = | A | + | B | + | C | - | A ∩ B | - | B ∩ C | - | A ∩ C | + | A ∩ B ∩ C |
        # |∪_{i=1}^{n} A_{i} | = \sum_{\phi \ne S \subset [n] (-1) ^{|S| + 1} |∩_{i \in S} A_{i}|

        # brute force Time: O(N * 2 ^ N)  space: O(N)
        from functools import reduce

        def interscet(A, B):
            return [f(a, b) for f, a, b in zip([max, max, min, min], A, B)]

        def area(rect):
            dx = max(0, rect[2] - rect[0])
            dy = max(0, rect[3] - rect[1])
            return dx * dy

        res = 0
        for n in range(1, len(rectangles) + 1):
            for group in itertools.combinations(rectangles, n):
                res += (-1) ** (n + 1) * area(reduce(interscet, group))
                pass

        return res % int(1e9 + 7)


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return '({:<},{:>})'.format(self.x, self.y)


class Solution_812:

    # 凸包 TODO this method also solved 587
    @staticmethod
    def convex_hull(points):

        def ccw(p1, p2, p3):
            # a = p3 - p1
            # b = p2 - p1
            # a x_train b = |a|*|b|sin<a,b> > 0 则b在a的逆时针方向
            return (p3[0] - p1[0]) * (p2[1] - p1[1]) - (p2[0] - p1[0]) * (p3[1] - p1[1]) > 0
            # return (p3.x-p1.x) * (p2.y - p1.y) - (p2.x-p1.x) * (p3.y - p1.y) > 0

        points = sorted(points)
        res = []
        for p in points:
            while len(res) >= 2 and ccw(res[-2], res[-1], p):
                res.pop()
            res.append(p)

        size = len(res)
        for p in points[::-1]:
            while len(res) - size >= 1 and ccw(res[-2], res[-1], p):
                res.pop()
            res.append(p)
        return res

    def largestTriangleArea(self, points):
        def area(A, B, C):
            return abs(A[0] * B[1] + B[0] * C[1] + C[0] * A[1] - A[1] * B[0] - B[1] * C[0] - C[1] * A[0]) * 0.5

        hull = self.convex_hull(points)
        m = 0
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                for k in range(j + 1, len(hull)):
                    m = max(m, area(hull[i], hull[j], hull[k]))
        return m


class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        c = 0
        i = len(num1) - 1
        j = len(num2) - 1
        res = []
        while i >= 0 and j >= 0:
            a = int(num1[i]) + int(num2[i]) + c
            s = a // 10
            c = a % 10
            res.append(str(s))
        while i >= 0:
            a = int(num1[i]) + c
            s = a // 10
            c = a % 10
            res.append(str(s))
        while j >= 0:
            a = int(num2[j]) + c
            s = a // 10
            c = a % 10
            res.append(str(s))
        if c:
            res.append(str(c))
        res.reverse()
        return ''.join(res)


class Solution_1042:
    def gardenNoAdj(self, N: int, paths):
        res = [0] * N
        graph = [[0 for _ in range(N)] for _ in range(N)]
        for x, y in paths:
            graph[x - 1][y - 1] = 1
            graph[y - 1][x - 1] = 1

        def dfs(i):
            nonlocal res
            to_use = {1, 2, 3, 4}
            for j in range(N):
                if graph[i][j] == 1 and res[j] != 0:
                    to_use -= {res[j]}
            if len(to_use) == 0:
                return
            res[i] = to_use.pop()
            for j in range(N):
                if graph[i][j] == 1 and res[j] == 0:
                    dfs(j)

        for i in range(N):
            if res[i] == 0:
                dfs(i)
        return res


class Solution_994:
    def orangesRotting(self, grid) -> int:
        m, n = len(grid), len(grid[0])

        d = 0
        que = []
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == 2:
                    que.append((i, j, 0))

        while que:
            x, y, d = que.pop(0)
            for nx, ny in [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = 2
                    que.append((nx, ny, d + 1))
        if any(1 in row for row in grid):
            return -1
        return d


class Solution_1095:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        n = mountain_arr.length()
        # 找峰值点
        left, right = 0, n - 1
        while left < right:
            mid = (left + right) // 2
            at_mid = mountain_arr.get(mid)
            if at_mid < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        ans = -1
        # 查找左半部分，升序
        i, j = 0, left
        while i <= j:
            m = (i + j) // 2
            atm = mountain_arr.get(m)
            if atm == target:
                ans = m
                break
            elif atm < target:
                i = m + 1
            else:
                j = m - 1

        if ans != -1:
            return ans
        # 查找右半部分，降序
        i, j = left + 1, n - 1
        while i <= j:
            m = (i + j) // 2
            atm = mountain_arr.get(m)
            if atm == target:
                return m
            elif atm > target:
                i = m + 1
            else:
                j = m - 1
        return -1


class Solution_1221:
    def balancedStringSplit(self, s: str) -> int:
        if len(s) == 0:
            return 0
        r = 0
        i = 0
        while i < len(s):
            t = s[i]
            cnt = 1
            i += 1
            while i < len(s) and s[i] == t:
                cnt += 1
                i += 1

            while i < len(s) and s[i] != t and cnt > 0:
                cnt -= 1
                i += 1
            if cnt == 0:
                r += 1
        return r


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __repr__(self):
        return f'{self.val}, {{next:{self.next}}}'


class Solution_937:
    def reorderLogFiles(self, logs):
        # TODO 一个特别 fancy 的想法，使用tuple来排序，然后sorted和sorted 都是稳定的
        def foo(log):
            _id, c = log.split(' ', 1)
            return (0, c, _id) if c[0].isalpha() else (1,)

        return sorted(logs, key=foo)


class Solution_925:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        sn = [''.join(list(v)) for _, v in itertools.groupby(name)]
        st = [''.join(list(v)) for _, v in itertools.groupby(typed)]
        if len(sn) != len(st):
            return False
        for n, t in zip(sn, st):
            if len(n) > len(t):
                return False
        return True


#######################################################################################################################
# 链表+双指针问题

class Solution_287:
    def findDuplicate(self, nums) -> int:
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[fast]
            fast = nums[fast]
            if slow == fast:
                break
        # 假设 循环链为  A --->  B ---> C
        #                       ^      |
        #                       +------+
        # 其中循环的入口点为B，假设快慢指针相遇点是C(C是环上任意一点)
        # 第一步，快慢指针同时出发
        #         慢指针：A -> B -> C 共N步
        #         快指针：A -> B -> ...->C 共2*N步
        # 第二步，新指针和慢指针同时不同地出发
        #         慢指针：C -> B -> C 共N步（又走了N步，相当于快指针，所以会到达C）
        #         新指针：A -> B -> C 共N步 (从开头出发，走N步到达C, )
        #        由于过程中两人同时经过B,因此一定在B点相遇。B->...->C长度是一样的，总长度一样
        ptr1 = nums[0]
        ptr2 = slow
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
        return ptr1

    def _findDuplicate(self, nums) -> int:
        index = nums[0]
        # 将对应位置设置为对应的数值，标记为已经访问过，如果已经访问过 index == nums[index] 就找到了环入口
        while index != nums[index]:
            index, nums[index] = nums[index], index
        return index


class Solution_142:
    def detectCycle(self, head: ListNode) -> ListNode:
        dummy = ListNode(None)
        dummy.next = head

        p = q = head
        while q:
            q = q.next
            if not q:
                break
            q = q.next
            p = p.next
            if p == q:
                break
        if not q:
            return None
        # print(p.val, q.val)
        t = head
        while p and p != t:
            p = p.next
            t = t.next
        return t


#######################################################################################################################

class Solution_28:
    def strStr(self, haystack: str, needle: str) -> int:
        # TODO: Boyer Moore算法
        def pre_bad_ch(p):
            m = len(p)
            bmbc = [m] * 128
            for i in range(m - 1):
                bmbc[ord(p[i])] = m - 1 - i
            return bmbc

        def pre_good_suffix(p):

            def suffix_brute_force(pattern):
                m = len(pattern)
                suff = [0] * m
                suff[m - 1] = m
                for i in range(m - 2, -1, -1):
                    j = i
                    while j >= 0 and pattern[j] == pattern[m - 1 - i + j]:
                        j -= 1
                    suff[i] = i - j
                print(suff)
                return suff

            def suffix(pattern):
                m = len(pattern)
                suff = [0] * m
                suff[m - 1] = m
                f = 0
                j = m - 1
                for i in range(m - 2, -1, -1):
                    if i > j and suff[i + m - 1 - f] < i - j:
                        suff[i] = suff[i + m - 1 - f]
                    else:
                        # j = min(j, i)
                        if i < j:
                            j = i
                        f = i
                        while j >= 0 and pattern[j] == pattern[j + m - 1 - f]:
                            j -= 1
                        suff[i] = f - j
                return suff

            m = len(p)
            suff = suffix_brute_force(p)
            bmgs = [m] * m
            j = 0
            for i in range(m - 1, -1, -1):
                if suff[i] == i + 1:
                    while j < m - 1 - i:
                        if bmgs[j] == m:  # 只记一次
                            bmgs[j] = m - 1 - i
                        j += 1
            for i in range(m - 2 + 1):
                bmgs[m - 1 - suff[i]] = m - 1 - i
            return bmgs

        def match(text, pattern):
            bmbc = pre_bad_ch(pattern)
            bmgs = pre_good_suffix(pattern)
            print('bc', bmbc[97:])
            print('gs', bmgs)
            j = 0
            while j <= len(text) - len(pattern):
                i = len(pattern) - 1
                while i >= 0 and pattern[i] == text[i + j]:
                    i -= 1
                if i < 0:
                    return j
                else:
                    j += max(bmbc[ord(text[i + j])] - len(pattern) + 1 + i, bmgs[i])
            return -1

        return match(haystack, needle)

    def KMP_strStr(self, haystack, needle):
        """
        KMP算法的核心是求 部分匹配表(Partial Match Table)数组，PMT中的值是字符串的前缀集合与后缀集合的交集中最长元素的长度
            具体使用下列例子说明：
                          index   0   1   2   3   4   5   6   7
                          char    a   b   a   b   a   b   c   a
                          PMT     0   0   1   2   3   4   0   1
                          next    -1  0   0   1   2   3   4   0   1

            首先，PMT数组的长度和模式串的长度相同, 上述例子中，
                  index=0，串 a     前缀集合{}， 后缀集合{}，交集为{}，故PMT[0] = 0
                  index=1, 串 ab    前缀集合{a}， 后缀集合{b}, 交集为{}，故PMT[1] = 0
                  index=2, 串 aba   前缀集合{a, ab}, 后缀集合{b, ba}, 交集为{a}, 故PMT[1] = 0
                                      ······
                  index=7, 串 abababca 前缀集合后缀集合交集为{a}, PMT[7] = 1

            匹配例子：
                       a     b    _a_   _b_   _a_   _b_   [a]   b   c   a     i = 6
                      _a_   _b_   _a_   _b_    a     b    [c]   a             j = 6
              在i=6,j=6时首先出现失配，而此时[i-j, i-1] 与[0, j-1]部分是已经匹配过完全相同的，而前面部分模式串中有相同的
              前缀和后缀 abab ,长度为PMT[j-1] = 4, 因此这一部分不需要再匹配，直接移动到 PMT[j-1] 与 i 匹配即可（i不需要
              移动）

         实际编程中，为了方便，常将 PMT 数组右移一位，0 位置设为 -1（编程方便），记为 next 数组，当出现失配时，移动到
         next[j] 进行匹配。

         求 next 数组（或者PMT数组）是找模式串前缀与后缀匹配的最长的长度，因此也是一个字符串匹配过程，即以模式字符串为主
         字符串，以模式字符串的前缀为目标字符串，一旦字符串匹配成功，那么当前的 next 值就是匹配成功的字符串的长度。匹配时，
         作为目标的模式串 index[1, len-1], 作为模式的模式串 index[0, len-2], 即保证前缀与后缀匹配
        """

        def cal_next(p):
            next = [0] * (len(p) + 1)
            next[0] = -1
            i = 0
            j = -1  # 初始化i=1,j=-1这样从模式串(i=1, j=0)开始匹配（会执行一遍i++,j++）
            while i < len(p):
                if j == -1 or p[i] == p[j]:
                    i += 1
                    j += 1
                    next[i] = j
                else:
                    j = next[j]
            return next

        def kmp(text, p):
            i = j = 0
            next = cal_next(p)
            while i < len(text) and j < len(p):
                if j == -1 or text[i] == p[j]:
                    i += 1
                    j += 1
                else:
                    j = next[j]
            if j == len(p):
                return i - j
            return -1

        return kmp(haystack, needle)

    def RabinKarp_strStr(self, text, pattern):
        def check(s, t):
            if len(s) != len(t):
                return False
            for i in range(len(s)):
                if s[i] != t[i]:
                    return False
            return True

        n, m = len(text), len(pattern)
        if n < m:
            return -1
        if m == 0:
            return 0

        MOD = int(1e9 + 7)
        p = 113
        h = p ** (m - 1) % MOD
        code = lambda c: ord(c) - ord('a')

        target = cur_val = 0
        for i in range(m):
            target = (target * p + code(pattern[i])) % MOD
            cur_val = (cur_val * p + code(text[i])) % MOD

        if cur_val == target and check(text[:m], pattern):
            return 0

        for i in range(m, n):
            cur_val = ((cur_val - code(text[i - m]) * h) * p + code(text[i])) % MOD
            # hash值相同，检查串是否相同
            if cur_val == target:
                if check(pattern, text[i - m + 1:i + 1]):
                    return i - m + 1
        return -1


class Solution_880:
    def decodeAtIndex(self, S: str, K: int) -> str:
        size = 0
        # 求出 解码后的 string 的 size
        for ch in S:
            if ch.isdigit():
                size *= int(ch)
            else:
                size += 1
        # 反向查找， 'apple' * 6, K = 24, ==> apple[K==4] 一样的
        # 所以可以 K %= size
        for ch in reversed(S):
            K %= size
            if K == 0 and ch.isalpha():
                return ch  # return at here
            if ch.isdigit():
                size //= int(ch)
            else:
                size -= 1


class Solution_459:
    def _repeatedSubstringPattern(self, s: str) -> bool:
        def check(a, s):
            if s == a:
                return True
            elif s[:len(a)] != a:
                return False
            else:
                return check(a, s[len(a):])

        for i in range(len(s) // 2):
            if check(s[:i + 1], s):
                return True
        return False

    def repeatedSubstringPattern(self, s: str) -> bool:
        # KMP solution
        if len(s) <= 1:
            return False
        _next = [0] * (len(s) + 1)
        _next[0] = -1  # 计算next数组，不能忘记这里
        i = 0
        j = -1
        while i < len(s):
            if j == -1 or s[i] == s[j]:
                i += 1
                j += 1
                _next[i] = j
            else:
                j = _next[j]

        x = _next[-1]
        print(_next)
        if x == 0:
            return False  # _next[-1] == 0 means s[1:] and s[:n-1] have no common substring
        # _next[-1] 代表 s[1:] 和 s[:n-1] 的最长公共子串（前后缀集合交集的最长值），则我们匹配s开头与剩余部分是否相同，
        # 如果相同则有：
        #               A  [A] [A] [A]
        #              [A] [A] [A]  A
        # A表示一个子串，[A] [A] [A] 表示s[1:] 和 s[:n-1] 的最长公共子串，而上下两个字符串是同一个，对应位置必然相同
        # 因此，S由A组成。
        for i in range(len(s) - x):
            if s[i] != s[i + x]:
                return False
        return True


class Solution_233:
    # link to 338 数 bit, 不太一样
    def countDigitOne(self, n: int) -> int:
        if n <= 0:
            return 0
        # 依次计算每一位上 '1' 可能出现的次数
        # 对于 个位            1,
        #                (1-9)1, say [11, 21, 31, 41, 51, 61, 71, 81, 91]
        #           (1-9)(0-9)1, say [101,...,191, 201, ......, 991]
        #            .......
        # 显然对于个位上可能出现1的次数为: 左边部分 + n % 10 != 0

        # 对于 十位           1(0-9), say [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        #               (1-9)1(0-9), say [110,..., 119, 120, ......., 919]
        #               .......
        # 对于十位上可能出现 1的次数为:  如果十位数字大于1：
        #                                 (左边部分 + 1) * 10
        #                             如果十位数字等于1：
        #                                 左边部分 * 10 + (右边部分 + 1) // 右边部分加1是因为可以取 [0,右边部分]闭区间
        #                             如果十位数字等于0：
        #                                 左边部分 * 10
        # 将个位的情况重新组织成：      如果个位数字大于1：
        #                                 (左边部分 + 1) * 1
        #                             如果个位数字等于1：
        #                                 左边部分 * 1 + (0 + 1)        // 个位时右边部分取0
        #                             如果十位数字等于0：
        #                                 左边部分 * 1
        # 即可归纳如下算法。
        ans = 0
        e = 1

        while n // e:
            left = n // (e * 10)
            right = n % e

            cur = n // e % 10
            if cur == 0:
                ans += left * e
            elif cur == 1:
                print(right, n - left * 10 * e)
                ans += left * e + right + 1
            else:
                ans += (left + 1) * e
            e *= 10
        return ans

    def countDigitOne(self, n: int) -> int:
        def countOne(s):
            if not s:
                return 0
            left_most_digit = int(s[0])  # 最高位
            n = len(s)  # 位数

            if n == 1 and left_most_digit >= 0:
                return 1 if left_most_digit > 0 else 0

            ans = 0
            if left_most_digit > 1:  # 最高位 > 1 则出现 10^(n-1) 方次
                ans += pow(10, n - 1)
            elif left_most_digit == 1:  # 最高位 ==1 出现右边的数字 + 1 次
                ans += int(s[1:]) + 1  # right + cur 的一个 1

            ans += left_most_digit * (n - 1) * pow(10, n - 2)  # 剩余 n-1 位中出现 1 的次数
            # left_most_digit * C(n-1, 1) * 10^(n-2)
            ans += countOne(s[1:])  # 递归
            return ans

        return countOne(str(n))


class Solution_149:
    def maxPoints(self, points) -> int:
        def gcd(a, b):
            while b > 0:
                a, b = b, a % b
            return a

        def line(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            A = y1 - y2
            B = x2 - x1
            # 不可能A,B都为零
            if A == 0:
                B = 1
            elif B == 0:
                A = 1
            else:
                if A < 0:
                    A, B = -A, -B
                r = gcd(abs(A), abs(B))
                A //= r
                B //= r
            return A, B

        res = 0
        for i, p1 in enumerate(points):
            cur = 1
            memo_slope = {}
            for j in range(1 + i, len(points)):
                p2 = points[j]
                if p1 == p2:
                    cur += 1
                else:
                    A, B = line(p1, p2)
                    memo_slope[(A, B)] = memo_slope.get((A, B), 0) + 1
            res = max(res, cur)
            if len(memo_slope):
                res = max(res, cur + max(memo_slope.values()))
        return res

    def _maxPoints(self, points) -> int:
        if len(points) < 3:
            return len(points)

        def gcd(a, b):
            while b > 0:
                a, b = b, a % b
            return a

        def line(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            res = [A, B, C]
            abst = [abs(A), abs(B), abs(C)]
            to = []
            for i in range(3):
                if abst[i] > 0:
                    to.append(i)
            if len(to) == 3:
                rt = gcd(abst[0], abst[1])
                r = gcd(rt, abst[2])
                print(abst, )
                for i in range(3):
                    res[i] //= r
            elif len(to) == 2:
                i, j = to
                r = gcd(abst[i], abst[j])
                res[i] //= r
                res[j] //= r
            elif len(to) == 1:
                res[to[0]] = 1
            else:
                pass
            return tuple(res)

        ps = {}
        m = 0
        for p in points:
            ps[tuple(p)] = ps.get(tuple(p), 0) + 1
            m = max(m, ps[tuple(p)])

        points = list(ps.keys())
        memo = {}
        for i, p1 in enumerate(points):
            for j in range(i + 1, len(points)):
                p2 = points[j]
                a, b, c = line(p1, p2)
                if (a, b, c) in memo:
                    memo[(a, b, c)].add(p1)
                    memo[(a, b, c)].add(p2)
                elif (-a, -b, -c) in memo:
                    memo[(-a, -b, -c)].add(p1)
                    memo[(-a, -b, -c)].add(p2)
                else:
                    memo[(a, b, c)] = {p1, p2}

        for k, v in memo.items():
            # print(k, v)
            r = 0
            for p in v:
                r += ps[p]
            m = max(m, r)
        return m


class Solution_65:
    def isNumber(self, s: str) -> bool:

        def isInteger(x, can_empty=False, cannot_sign=False):
            c = 0
            while len(x) > 0 and (x[0] in ['-', '+']):
                if cannot_sign:
                    return False
                x = x[1:]
                c += 1

            if c > 1 or (not can_empty and len(x) == 0):
                return False

            for c in x:
                if not c.isdigit():
                    return False
            return True

        def isDecimal(x):
            t = x.split('.')
            if len(t) == 1:
                return isInteger(t[0])
            elif len(t) == 2:
                if len(t[0].strip('-').strip('+')) == len(t[1].strip('-').strip('+')) == 0:
                    return False
                left = isInteger(t[0], can_empty=True)
                right = isInteger(t[1], can_empty=True, cannot_sign=True)
                return left and right
            else:
                return False

        def isScience(x):
            t = x.split('e')
            if len(t) == 1:
                return isDecimal(t[0])
            elif len(t) == 2:
                return isDecimal(t[0]) and isInteger(t[1])
            else:
                return False

        s = s.strip(' ')
        return isScience(s)


#######################################################################################################################
# 单调栈
# 使用stack 保持一个升序/降序的序列来解题，每一个元素都入栈一次。
# 第 84 题同时还写了线段树解法。
# 单调栈，用来维护一个单调的序列（如升序），一般的范式如下：
# >>> stack = []
# >>> for x in nums:
# >>>     while stack and stack[-1] > x:
# >>>         stack.pop()
# >>>     stack.append(x)
# 这样栈中存储的就是一个单调序列
class Solution_907:
    def sumSubarrayMins(self, A: List[int]) -> int:
        # 要找出 A 的所有子集的最小元素的和，暴力是 O(2^n)
        # 关键在于计算 A 中每个元素至少在多少个子集中为最小值，设 A[i] 在 f(i) 个子集中为最小元素
        # 则有 ans = \sum_i (A[i] * f(i)), 于是问题转变为求 f(i)
        #
        # 单调栈可以用来求前一大/小元素或者是后一大/小元素，我们可以利用这个性质来求 f(i)
        # 先看一个例子，[2, 8, 7, 3, 4, 6, 9, 1]
        #               ^        ^           ^
        # 显然，元素 3 的前一小元素是 2，后一小元素是 1
        # 我们用 left[i] 记录元素 A[i] 到它的前一小元素的距离，这里 3 的 index=3, left[3] = 3
        # 我们用 right[i] 记录元素 A[i] 到它的后一小元素的距离，这里 3 的 index=3, right[3] = 4
        # 于是，包含元素 A[4] = 3 的子集有。
        #           3           # 只有 3
        #        7, 3           # 只有左边
        #     8, 7, 3
        #           3, 4        # 只有右边
        #           3, 4, 6
        #           3, 4, 6, 9
        #        7, 3, 4        # 左右都有
        #        7, 3, 4, 6
        #        7, 3, 4, 6, 9
        #     8, 7, 3, 4
        #     8, 7, 3, 4, 6
        #     8, 7, 3, 4, 6, 9
        # 共 12 个，即 f(4) = 12, 对应:
        #   f(4) = 1 + (left[4] - 1) + (right[4] - 1) + (left[4] - 1) * (right[4] - 1)
        #        = left[4] * right[4]
        # 即有 f(i) = left[i] * right[i]
        # 而使用单调栈可以很容易的求出 left 和 right 数组。

        n = len(A)
        # 初始化
        left, right = [i + 1 for i in range(n)], [n - i for i in range(n)]
        # 求到前一个小的元素的距离
        stack = []
        for i, x in enumerate(A):
            while stack and A[stack[-1]] > x:
                stack.pop()
            left[i] = i - stack[-1] if stack else i + 1
            stack.append(i)
        # 求到后一个小的元素的距离
        # 注意求前一个小的元素是严格的小于，而求后一个小于是小于等于，
        # 体现在上下两段代码中就是两端代码中就是对 left 和 right 赋值的过程，
        # 一个是在判断循环内，一个在判断循环外，而比较用到都是大于号（A[stack[-1]] > x）
        stack = []
        for i, x in enumerate(A):
            while stack and A[stack[-1]] > x:
                t = stack.pop()
                right[t] = i - t
            stack.append(i)

        ans = 0
        MOD = int(1e9 + 7)
        for x, l, r in zip(A, left, right):
            ans = (ans + x * l * r) % MOD
        return ans

    def sumSubarrayMins(self, A: List[int]) -> int:
        # 注意上面代码中，求 left 和 right 的逻辑是一样的，仅仅是赋值的时候不同，
        # 所以可以将其合并到一个循环中。
        # 1). right 的赋值是在 stack 的循环内进行的, 每次对 pop 出来下标对应的元素计算 right 所以不用改动。
        # 2). left 的赋值在上面的解法中是按顺序进行，我们如果对每次 pop 出来下标对应元素计算 left 应该怎么算呢？
        #     很简单，栈式升序有序的，弹出后下一个就是栈顶就是弹出元素的前一小，当然如果栈为空了就是弹出元素的下标加一
        # 最后，既然求 left, right 可以在 one-pass 做完，那么求最后结果也可以合并进来，最后的代码如下
        ans = 0
        MOD = int(1e9 + 7)
        n = len(A)
        stack = []
        for i in range(n + 1):
            while stack and A[stack[-1]] > (A[i] if i < n else 0):
                t = stack.pop()
                k = stack[-1] if stack else -1
                ans += A[t] * (i - t) * (t - k)  # right[t] = i - t, left[t] = t - k
            stack.append(i)
        return ans % MOD


class SegmentTree:
    def __init__(self, ls):
        k = math.ceil(math.log(len(ls), 2))  # 线段树深度
        max_size = 2 * pow(2, k) - 1  # 线段树最大节点数
        st = [0] * max_size

        def _construct(lo, hi, idx):
            nonlocal st
            if lo == hi:
                st[idx] = lo
            else:
                mid = lo + (hi - lo) // 2
                st[idx] = self.min_val(ls, _construct(lo, mid, idx * 2 + 1), _construct(mid + 1, hi, idx * 2 + 2))
            return st[idx]

        _construct(0, len(ls) - 1, 0)
        self.ls = ls
        self.st = st

    @staticmethod
    def min_val(arr, i, j):
        if i == -1:
            return j
        elif j == -1:
            return i
        else:
            return i if arr[i] < arr[j] else j

    def RMQ(self, lo, hi):
        n = len(self.ls)

        def _rmq(low, high, qlo, qhi, idx):
            if high < qlo or low > qhi:
                return -1
            elif qlo <= low and qhi >= high:
                return self.st[idx]
            mid = low + (high - low) // 2
            return self.min_val(self.ls, _rmq(low, mid, qlo, qhi, 2 * idx + 1),
                                _rmq(mid + 1, high, qlo, qhi, 2 * idx + 2))

        if 0 <= lo <= hi < n:
            return _rmq(0, n - 1, lo, hi, 0)
        else:
            return -1


class Solution_84:
    def largestRectangleArea(self, heights: 'List[int]') -> int:
        # 使用线段树
        st = SegmentTree(heights)

        def maxArea(l, r):
            if l > r:
                return -float('inf')
            elif l == r:
                return st.ls[l]
            m = st.RMQ(l, r)
            return max(maxArea(l, m - 1), maxArea(m + 1, r), (r - l + 1) * st.ls[m])

        return maxArea(0, len(st.ls) - 1)

    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        n = len(heights)
        ans = 0
        for i in range(n):  # 注意循环次数
            while stack and heights[stack[-1]] > heights[i]:
                t = stack.pop()
                k = stack[-1] if stack else -1
                area = heights[t] * (i - 1 - k)
                ans = max(ans, area)
            stack.append(i)

        while stack:
            t = stack.pop()
            area = heights[t] * (n - 1 - stack[-1] if stack else n)
            ans = max(ans, area)

        return ans

    def largestRectangleArea(self, heights: List[int]) -> int:
        # 这个写法和 907 题的 one-pass 解法几乎是一样的
        # 找出每个数的前一小元素与后一小元素，然后面积是左右之间的距离乘于当前块的高度
        # 下面 ans = max(ans, heights[t] * ((i - t) + (t - k) - 1)) 一句中没有化简求和式是为了和上面保持一致
        stack = []
        ans = 0
        n = len(heights)
        heights.append(0)
        for i in range(n + 1):  # 注意循环次数多了 1 次
            # 循环多了 i==n 的情况，同时判断 height[stack[-1]] > 0 (heights 全为非负数)
            # 是将上面解法中 最后的判断 stack 内的合并到了一起。
            while stack and heights[stack[-1]] > heights[i]:
                t = stack.pop()
                k = stack[-1] if stack else -1
                ans = max(ans, heights[t] * ((i - t) + (t - k) - 1))
            stack.append(i)
        return ans


class Solution_85:
    # stack, dp
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # 这个题目可以变形为 84. Largest Rectangle in Histogram 问题
        # 我们列中的 '1' 看作一个单位的小方格，然后维护一个 heights 数组，
        # 这个数组表示扫描到当前位置时累积的每一列的高度，因为要求的是矩形
        # 是一个连续区域，于是我们在遇到 '0' 时 reset 当前列的 heights 值。
        # 举例, 一个 matrix 如下：
        #     [["1", "0", "1", "0", "0"],
        #      ["1", "0", "1", "1", "1"],
        #      ["0", "1", "1", "1", "1"],
        #      ["1", "0", "0", "1", "0"]]
        # 扫描第一行后 heights = [1, 0, 1, 0, 0]
        # 扫描第二行后 heights = [2, 0, 2, 1, 1]
        # 扫描第三行后 heights = [0, 1, 3, 2, 2]  注意第 0 列的高度被重置为 0
        # 扫描第四行后 heights = [1, 0, 0, 3, 0]  注意第 1, 2, 4 列高度被重置为 0
        #
        # 每扫描一个元素，我们都更新一次 heights(上面只列出来扫描完一行后的)，更新 heights 的同时
        # 我们可以解决 Largest Rectangle in Histogram 问题。最后得到的就是问题的解。

        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if m == 0 or n == 0:
            return 0

        res = 0
        heights = [0] * (n + 1)
        for i in range(m):
            stack = []
            for j in range(n + 1):
                # 更新 heights
                if j < n:
                    heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
                # 计算 面积
                while stack and heights[stack[-1]] >= heights[j]:
                    t = stack.pop()
                    k = stack[-1] if stack else -1
                    res = max(res, heights[t] * (j - k - 1))
                stack.append(j)
        return res


class Solution_739:
    def _dailyTemperatures(self, T: List[int]) -> List[int]:
        n = len(T)
        nexts = [n] * 72  # one more to avoid empty sequence for min
        res = []
        for i in range(n - 1, -1, -1):
            m = min(nexts[t] for t in range(T[i] - 30 + 1, 72))
            res.append(m - i if m < n else 0)
            nexts[T[i] - 30] = i
        return res[::-1]

    def _dailyTemperatures(self, T: List[int]) -> List[int]:
        # forward fashion, space O(n)
        stack = []
        res = [0] * len(T)
        for i in range(len(T)):
            while stack and T[stack[-1]] < T[i]:
                idx = stack.pop()
                res[idx] = i - idx
            stack.append(i)
        return res

    def dailyTemperatures(self, T: List[int]) -> List[int]:
        # backward fashion, space O(w), w ~ 70([30, 100])
        stack = []
        res = [0] * len(T)
        for i in range(len(T) - 1, -1, -1):
            while stack and T[i] >= T[stack[-1]]:
                stack.pop()
            if stack:
                res[i] = stack[-1] - i
            stack.append(i)
        return res


class Solution_496:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # brute force, AC but ugly
        res = []
        for x in nums1:
            idx = nums2.index(x)
            t = -1
            for i in range(idx + 1, len(nums2)):
                if nums2[i] > x:
                    t = nums2[i]
                    break
            res.append(t)
        return res

    def nextGreaterElement(self, findNums, nums):
        stack, d = [], {}
        # 用栈维护一个降序的序列，每次我们看到一个比栈顶元素大的元素x, 则弹出栈中
        # 所有比x小的元素，他们的NextGreaterElem就是x
        for x in nums:
            while stack and stack[-1] < x:
                d[stack.pop()] = x
            stack.append(x)

        return [d.get(x, -1) for x in findNums]


class Solution_503:
    # stack， 维护降序序列
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        # 循环数组，把数组自身拼接以下就好了
        res = [-1] * 2 * len(nums)
        stack = []
        tmp = nums + nums
        for i, x in enumerate(tmp):
            while stack and tmp[stack[-1]] < x:
                t = stack.pop()
                res[t] = x
            stack.append(i)
        return res[:len(nums)]

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        # 做两遍循环也是一样的
        res = [-1] * len(nums)
        stack = []
        for i in range(2):
            for i, x in enumerate(nums):
                while stack and nums[stack[-1]] < x:
                    t = stack.pop()
                    res[t] = x
                stack.append(i)
        return res


class Solution_402:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        cnt = 0
        for i, c in enumerate(num):
            while stack and ord(stack[-1]) > ord(c) and cnt < k:
                stack.pop()
                cnt += 1
            stack.append(c)
        while cnt < k and stack:
            stack.pop()
            cnt += 1
        res = ''.join(stack).lstrip('0')
        return res if res else '0'


class Solution_316:
    def removeDuplicateLetters(self, s: str) -> str:
        # 使用栈维护一个字符的升序序列(尽可能的升序，因为要保持原本的相对顺序)
        stack = []
        cnt = {}
        seen = {}
        for c in s:
            cnt[c] = cnt.get(c, 0) + 1
            seen[c] = False
        for c in s:
            cnt[c] -= 1  # cnt 表示后面还剩多少个每个字符
            if seen[c]:  # 栈里面已经有 c 了，则 c 一定处在一个相对升序序列中
                continue
            while stack and ord(stack[-1]) > ord(c) and cnt[stack[-1]] > 0:
                t = stack.pop()
                seen[t] = False
            stack.append(c)
            seen[c] = True
        return ''.join(stack)

    @staticmethod
    def debug():
        s = Solution_316()
        assert s.removeDuplicateLetters("abacb") == "abc"
        assert s.removeDuplicateLetters("cbacdcbc") == "acdb"
        assert s.removeDuplicateLetters("bcabc") == "abc"


class Solution_321:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:

        def pre(nums, k):
            drop = len(nums) - k  # number of num need to drop
            stack = []
            for x in nums:
                # 维护一个降序序列
                while stack and drop and stack[-1] < x:
                    stack.pop()
                    drop -= 1
                stack.append(x)
            return stack[:k]

        def merge(a, b):
            # a, b 都是 list max(a, b).pop(0) 从比较大的比较结果中 pop 头一个元素
            return [max(a, b).pop(0) for _ in a + b]

        m, n = len(nums1), len(nums2)
        ans = []
        # 枚举所有的情况
        for i in range(k + 1):
            if i <= m and k - i <= n:
                tmp = merge(pre(nums1, i), pre(nums2, k - i))
                ans = max(ans, tmp)
        return ans


class Solution_456:
    # stack, 维护有序序列，区间查找
    def find132pattern(self, nums: List[int]) -> bool:
        # TLE, O(n^3)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    if nums[i] < nums[k] < nums[j]:
                        return True
        return False

    def find132pattern(self, nums: List[int]) -> bool:
        # TLE, O(n^2)
        min_i = float('inf')
        for j in range(len(nums) - 1):
            min_i = min(min_i, nums[j])  # min_i 记录从 i ~ j 中最小的
            # 我们要找出一个 num[k] 落入到区间 (min_i, nums[j]) 中
            for k in range(j + 1, len(nums)):
                if min_i < nums[k] < nums[j]:
                    return True
        return False

    def find132pattern(self, nums: List[int]) -> bool:
        # TLE, O(n^2), 使用一个数据维护 (nums[i], nums[j]) 区间
        idx = 0
        intervals = []
        for i in range(len(nums)):
            if nums[i] <= nums[i - 1]:
                if idx < i - 1:
                    intervals.append((nums[idx], nums[i - 1]))
                idx = i
            for ai, aj in intervals:
                if ai < nums[i] < aj:
                    return True
        return False

    def find132pattern(self, nums: List[int]) -> bool:
        # O(n) 使用 stack 维护有序序列
        if len(nums) < 3:
            return False

        stack = []
        # mins 存储从 0 ~ i 区间的最小值
        mins = [0] * len(nums)
        mins[0] = nums[0]
        for i in range(1, len(nums)):
            mins[i] = min(mins[i - 1], nums[i])

        for j in range(len(nums) - 1, -1, -1):  # 注意这里是反向循环的
            if nums[j] > mins[j]:
                # 使用 stack 维护了一个严格的降序序列
                while stack and stack[-1] <= mins[j]:
                    stack.pop()
                # 如果 stack 中还有元素的话满足 stack[-1] > mins[j]
                if stack and stack[-1] < nums[j]:
                    return True
                stack.append(nums[j])
        return False

    def find132pattern(self, nums: List[int]) -> bool:
        # O(n) 把数组本身当作 stack 维护有序序列
        if len(nums) < 3:
            return False

        # mins 存储从 0 ~ i 区间的最小值
        mins = [0] * len(nums)
        mins[0] = nums[0]
        for i in range(1, len(nums)):
            mins[i] = min(mins[i - 1], nums[i])
        k = len(nums) - 1  # stack 的指针
        for j in range(len(nums) - 1, -1, -1):  # 注意这里是反向循环的
            if nums[j] > mins[j]:
                # 使用 stack 维护了一个严格的降序序列
                while k < len(nums) - 1 and nums[k + 1] <= mins[j]:
                    k += 1  # 出栈
                # 如果 stack 中还有元素的话满足 nums[k] > mins[j]
                if k < len(nums) - 1 and nums[k + 1] < nums[j]:
                    return True

                nums[k] = nums[j]  # 入栈（把nums本省当作栈，k-=1 以及 这一句赋值时入栈过程）
                k -= 1
        return False


class Solution_239:
    # 单调队列
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 暴力解 O(k*n)
        if len(nums) == 0:
            return []

        i = 0
        res = []

        while i + k <= len(nums):
            res.append(max(nums[i:i + k]))
            i += 1
        return res

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 对暴力解进行优化， 滑动
        if len(nums) == 0:
            return []
        res = [max(nums[:k])]
        i = 1
        while i + k <= len(nums):
            prev = res[-1]
            if prev < nums[i + k - 1]:
                res.append(nums[i + k - 1])
            elif prev > nums[i - 1]:
                res.append(prev)
            else:
                res.append(max(nums[i:i + k]))
            i += 1
        return res

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if len(nums) == 0:
            return []
        res = []
        que = collections.deque()
        for i, x in enumerate(nums):
            while que and nums[que[-1]] < x:
                que.pop()
            que.append(i)

            if que[0] == i - k:
                que.popleft()

            if i >= k - 1:
                res.append(nums[que[0]])
        return res


class StockSpanner:

    def __init__(self):
        self.stack = []
        self.idx = 1

    def next(self, price: int) -> int:

        while self.stack and self.stack[-1][1] <= price:
            self.stack.pop()
        if self.stack:
            r = self.idx - self.stack[-1][0]
        else:
            r = self.idx
        self.stack.append([self.idx, price])
        self.idx += 1
        return r


#######################################################################################################################

class Solution_769:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        # 划分点 左边的元素都大于/小于，右边的元素都小于/大于划分点
        # 这个类似于快排的划分
        # 然后这个题目就类似于 42 题，积水面积, 不过是要大于左边最大，小于右边最小，加一
        maxs = [-1]
        mins = [float('inf')]
        for x in arr:
            maxs.append(max(maxs[-1], x))

        for x in arr[::-1]:
            mins.append(min(mins[-1], x))
        maxs.append(float('inf'))
        mins.append(float('inf'))
        mins.reverse()
        ans = 1
        for i, x in enumerate(arr[1:]):
            # print(maxs[i], x, mins[i+2])
            if maxs[i + 1] < x < mins[i + 3]:
                ans += 1
        return ans


class Solution_212:
    def findWords(self, board, words):
        m, n = len(board), len(board[0])
        visit = [[0] * n for _ in range(m)]
        res = set()

        def create_trie(words):
            root = {}
            for i, w in enumerate(words):
                p = root
                for c in w:
                    if c not in p:
                        p[c] = {}
                    p = p[c]
                p['#'] = i
            return root

        def dfs(i, j, trie):
            nonlocal visit, res
            if '#' in trie:
                res.add(trie['#'])

            for x, y in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                ni, nj = x + i, y + j
                if 0 <= ni < m and 0 <= nj < n and (board[ni][nj] in trie) and visit[ni][nj] == 0:
                    visit[ni][nj] = 1
                    dfs(ni, nj, trie[board[ni][nj]])
                    visit[ni][nj] = 0

        root = create_trie(words)
        for i in range(m):
            for j in range(n):
                if board[i][j] in root:
                    visit[i][j] = 1
                    dfs(i, j, root[board[i][j]])
                    visit[i][j] = 0
        return [words[i] for i in res]


######################################################################################################################
import heapq


class Solution_218:
    # TODO 看不懂
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # left:(l,r,height) right:(r,infinity,0)
        events = sorted(buildings + [[b[1], float("inf"), 0] for b in buildings])
        res = []
        # heap: (-height, right)
        heap = [(0, float("inf"))]
        for l, r, h in events:
            top = heap[0][0]
            while heap[0][1] <= l:
                heapq.heappop(heap)
            if h > 0:
                heapq.heappush(heap, (-h, r))
            if top != heap[0][0]:
                res.append([l, -heap[0][0]])
        return res


######################################################################################################################
class Solution_313:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        res = {1}
        while len(res) < n:
            tmp = set()
            print(res)
            for x in res:
                for p in primes:
                    tmp.add(x * p)
            print('#', tmp)
            tmp = tmp - res
            res.add(min(tmp - res))
        print(sorted(res))
        return sorted(list(res))[n - 1]


# print(Solution_313().nthSuperUglyNumber(12, [2, 7, 13, 19]))


class Solution_179:
    def _largestNumber(self, nums: List[int]) -> str:
        from functools import cmp_to_key
        def cmp(a, b):
            i = j = 0
            cnt = 0
            reach_a = reach_b = False
            while not reach_a or not reach_b:
                ca = a[i]
                cb = b[j]
                if ca > cb:
                    return 1
                elif ca < cb:
                    return -1
                else:
                    if reach_b:
                        nj = (j + 1) % len(b)
                        if b[j] < b[nj]:
                            return -1
                        elif b[j] > b[nj]:
                            return 1
                    elif reach_a:
                        ni = (i + 1) % len(a)
                        if a[i] < a[ni]:
                            return 1
                        elif a[i] > a[ni]:
                            return -1
                cnt += 1
                reach_a = cnt >= len(a)
                reach_b = cnt >= len(b)
                i = (i + 1) % len(a)
                j = (j + 1) % len(b)
            return 0

        t = [str(x) for x in nums]

        t.sort(key=cmp_to_key(cmp), reverse=True)
        res = ''.join(t).lstrip('0')
        if len(res) == 0:
            return '0'
        return res

    def largestNumber(self, nums: List[int]) -> str:
        from functools import cmp_to_key
        # TODO this is way more better!!!!!!!!!!
        def cmp(a, b):
            if (a + b) < (b + a):
                return -1
            elif (a + b) > (b + a):
                return 1
            return 0

        t = [str(x) for x in nums]

        t.sort(key=cmp_to_key(cmp), reverse=True)
        # print(t)
        res = ''.join(t).lstrip('0')
        if len(res) == 0:
            return '0'
        return res


class Solution_796:
    def _rotateString(self, A: str, B: str) -> bool:
        return len(A) == len(B) and B in A + A

    def _rotateString(self, A, B):
        # brute force
        if A == B:
            return True
        i = 0
        ls = list(A)
        while i < len(A):
            if ''.join(ls) == B:
                return True
            t = ls.pop(0)
            ls.append(t)
            i += 1
        return False

    def rotateString(self, A, B):
        # Rolling Hash
        MOD = int(1e9 + 7)
        P = 113
        Pinv = pow(P, MOD - 2, MOD)

        def get_hash(string):
            hs = 0
            power = 1
            for x in string:
                code = ord(x) - 96  # since ord('a') = 97
                hs = (hs + power * code) % MOD
                power = power * P % MOD
            return hs, power

        hb, power = get_hash(B)
        ha, power = get_hash(A)

        if ha == hb and A == B:
            return True
        for i, x in enumerate(A):
            code = ord(x) - 96
            ha += power * code
            ha -= code
            ha *= Pinv
            ha %= MOD
            if ha == hb and A[i + 1:] + A[:i + 1] == B:
                return True
        return False

    def _rotateString(self, A, B):
        # TODO KMP
        pass


class Solution_15:
    def threeSum(self, nums):
        # two pointers, 排序后搜索，O(n^2)
        # 对于升序的nums, j从前往后，k从后向前，如果大于目标，k向前移动，如果小于目标值，k相后移动
        res = set()
        nums.sort()

        for i in range(len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            a = nums[i]
            while j < k:
                b, c = nums[j], nums[k]
                if a + b + c > 0:
                    k -= 1
                elif a + b + c == 0:
                    res.add((a, b, c))
                    while j < k and nums[j] == nums[j + 1]: j += 1
                    while j < k and nums[k] == nums[k - 1]: k -= 1
                    k -= 1
                else:
                    j += 1
        return list(res)

    def threeSum(self, nums):
        res = []
        negative, zeros, positive = {}, 0, {}
        for x in nums:
            if x < 0:
                negative[x] = negative.get(x, 0) + 1
            elif x > 0:
                positive[x] = positive.get(x, 0) + 1
            else:
                zeros += 1
        if len(negative) and len(positive):
            if zeros > 0:
                for x in negative:
                    if -x in positive:
                        res.append([x, 0, -x])
            for a, b in itertools.combinations_with_replacement(negative.keys(), 2):
                if a == b and negative[a] < 2:
                    continue
                c = -(a + b)
                if c in positive:
                    res.append([a, b, c])
            for b, c in itertools.combinations_with_replacement(positive.keys(), 2):
                if b == c and positive[b] < 2:
                    continue
                a = -(b + c)
                if a in negative:
                    res.append([a, b, c])
        if zeros >= 3:
            res.append([0, 0, 0])
        return res


class Solution_5233:
    def _TLE_jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        tasks = list(zip(startTime, endTime, [-p for p in profit]))
        tasks = sorted(tasks)
        sss = [0] * len(tasks)
        sss[-1] = tasks[-1][2]
        for i in range(len(tasks) - 2, -1, -1):
            sss[i] = tasks[i][2] + sss[i + 1]
        m = 0

        def foo(idx, st, tasks, profit):
            nonlocal m
            m = min(m, profit)
            if idx >= len(tasks):
                return
            if tasks[idx][0] >= st:
                foo(idx + 1, tasks[idx][1], tasks, profit + tasks[idx][2])
            if sss[idx + 1] < m:
                foo(idx + 1, st, tasks, profit)

        foo(0, 1, tasks, 0)
        return -m

    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        # TODO unsolved
        tasks = list(zip(startTime, endTime, profit))
        tasks = sorted(tasks)
        dp = [[0] * len(tasks) for _ in range(len(tasks))]
        tp = [[0] * len(tasks) for _ in range(len(tasks))]
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if i > 0:
                    if tasks[tp[i - 1][j - 1]][1] <= tasks[i][0] and (dp[i - 1][j - 1] + tasks[i][2]) > dp[i][j]:
                        dp[i][j] = dp[i - 1][j - 1] + tasks[i][2]
                        tp[i][j] = i
                    else:
                        dp[i][j] = dp[i - 1][j]
        for ls in dp:
            print(ls)
        print('###')
        for ls in tp:
            print(ls)
        return dp[len(tasks) - 1][len(tasks) - 1]


#######################################################################################################################
# 下面几道题是二分的代表
# NOTE 二分法的应用关键在于，可行解的结构是单调的，且在一可确定的区间内。
#      可以通过构造改变求解的方式,构造关键字的比较方式，如1201题。
#      构造出对一个可能解的检查办法，确定其是否符合条件

class Solution_1201:
    def _TLE_nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        tojump = {}
        for i in [a, b, c, a * b, a * c, b * c]:
            tojump[i] = 1

        res = 0
        ia = ib = ic = 1
        i = 0
        while i < n:
            t = min(a * ia, b * ib, c * ic)
            # print(t, ia, ib, item_to_cate)

            if t in tojump and tojump[t] == 1:
                tojump[t] = 0
                res = t
                i += 1
            elif t not in tojump:
                res = t
                i += 1
            ia += (t == a * ia)
            ib += (t == b * ib)
            ic += (t == c * ic)

        return res

    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        # TODO 用到了最小公倍数（LCM)加上二分查找。
        from math import gcd

        def LCM(a, b, c=1):
            if c == 1:
                return (a * b) // gcd(a, b)
            lcm_ab = (a * b) // gcd(a, b)
            lcm_bc = (b * c) // gcd(b, c)
            return (lcm_ab * lcm_bc) // gcd(lcm_ab, lcm_bc)

        def f(k):
            """f(k) return the number of numbers that can be divide by a/b/c"""
            return k // a + k // b + k // c - k // LCM(a, b) - k // LCM(b, c) - k // LCM(a, c) + k // LCM(a, b, c=c)

        left = min(a, b, c)
        right = min(n * max(a, b, c), int(2e9 + 1))
        while left < right:
            mid = (right + left) // 2
            print(f'{mid}:{f(mid)} ', end=' ')
            if f(mid) < n:
                left = mid + 1
            else:
                right = mid
        print('')
        return left

    def _nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        def lcm(x, y):
            return x * y // math.gcd(x, y)

        def count_ugly(n, a, b, c, ab, bc, ca, abc):
            answer = n // a + n // b + n // c
            answer -= n // ab + n // bc + n // ca
            answer += n // abc
            return answer

        ab, bc, ca = lcm(a, b), lcm(b, c), lcm(c, a)
        abc = lcm(ab, c)
        low = 1
        high = 2 * 10 ** 9
        while low < high:
            mid = low + (high - low) // 2
            if count_ugly(mid, a, b, c, ab, bc, ca, abc) < n:
                low = mid + 1
            else:
                high = mid
        return low


class Solution_878:
    def nthMagicalNumber(self, N: int, A: int, B: int) -> int:
        # 这题和 1201. n-th ugly number III 其实是一样的，大致是那题的简化版

        from math import gcd
        MOD = int(1e9 + 7)
        lcm = A // gcd(A, B) * B  # 最小公倍数

        # 容斥原理
        def f(k):
            return k // A + k // B - k // lcm

        lo, hi = min(A, B), N * max(A, B)
        k = 0

        while lo < hi:
            mid = lo + (hi - lo) // 2
            t = f(mid)
            if t < N:
                lo = mid + 1
            else:
                hi = mid

        return lo % MOD


class Solution_410:
    def _splitArray(self, nums: List[int], m: int) -> int:
        # 结果的取值区间为： [max(nums), sum(nums)]
        # 可以在这个区间内进行二分查找
        # 给定一个检查的值max_sum, 确定数组能否分成m个 sum<= max_sum的子集
        # 能否分成m个，可以反过来检查不大于max_sum的子数组数目，如果超过m个则不能划分
        def can_split(max_sum):
            cur, cuts = 0, 0
            for x in nums:
                if cur + x > max_sum:
                    cuts += 1
                    cur = x
                else:
                    cur += x
            return cuts < m  # (cuts + 1) <= m 最后cur中还剩一个子集。

        lo = max(nums)
        hi = sum(nums)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if can_split(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def splitArray(self, nums, m):
        # TODO 这个dp怎么做的？
        n = len(nums)
        dp = [[-1] * (m + 1) for _ in range(n + 1)]

        def foo(i, j):
            nonlocal dp
            if dp[i][j] >= 0:
                return dp[i][j]
            if j == 1:
                return sum(nums[i:])
            sums = 0
            kk = float('inf')
            for k in range(i, n - j + 1):
                sums += nums[k]
                kk = min(kk, max(sums, foo(k + 1, j - 1)))
            dp[i][j] = kk
            return dp[i][j]

        return foo(0, m)


class Solution_198:
    def rob(self, nums: List[int]) -> int:
        last = now = 0
        for x in nums:
            last, now = now, max(last + x, now)
        return now


class Solution_740:
    def deleteAndEarn(self, nums: List[int]) -> int:
        # 这个题有点像 198. House Robber, 198 题的递推如下
        # last, now = now, max(last+x, now)
        # 这里用 Counter 将数值分桶，然后使用上面的方案

        count = collections.Counter(nums)
        prev = None
        last = now = 0
        for k in sorted(count.keys()):
            if k - 1 != prev:
                last, now = now, k * count[k] + now
            else:
                last, now = now, max(k * count[k] + last, now)
            prev = k
        return now

    def deleteAndEarn(self, nums: List[int]) -> int:
        # 由于 nums[i] 取值范围为 [1, 10000]
        # 我们可以使用 10001 的数组来分桶，这样可以避免 Counter 还要判断 key 的问题
        # 不过可能会慢一点
        bukets = [0] * 10001
        for x in nums:
            bukets[x] += x

        last = now = 0
        for x in bukets:
            last, now = now, max(last + x, now)
        return now


class Solution:
    def threeSumMulti(self, A: List[int], target: int) -> int:
        # n = len(A) \in [3, 3000] 暴力肯定不行
        # O(n^2) TLE
        # 思路是固定一个，然后转化成 2 sum(排序，然后双指针)
        # 排序并不影响，实际上我们要找的是 A[i] + A[j] + A[k] == target 且 i != j != k
        # 因此我们总是可以重新映射 i,j,k 使其满足 i<j<k
        A.sort()
        MOD = int(1e9 + 7)
        ans = 0
        for i in range(len(A)):
            T = target - A[i]
            # 典型的 two sum, 双指针法
            lo, hi = i + 1, len(A) - 1
            while lo < hi:
                if A[lo] + A[hi] < T:
                    lo += 1
                elif A[lo] + A[hi] > T:
                    hi -= 1
                elif A[lo] != A[hi]:
                    # 找到了一个 A[lo] + A[hi] == T 并且 A[lo] != A[hi]
                    # 我们把数 左边都等于 A[lo]的，然后数 右边都等于 A[hi] 的
                    # 计数分别是 left, right, 然后注意移动指针
                    left = right = 1
                    while lo + 1 < hi and A[lo] == A[lo + 1]:
                        left += 1
                        lo += 1
                    while hi - 1 > lo and A[hi] == A[hi - 1]:
                        right += 1
                        hi -= 1
                    ans += left * right
                    ans %= MOD
                    hi -= 1
                    lo += 1
                else:
                    # A[lo] == A[hi], 说明 lo~hi 之间全都是同一个数，总共 m = hi - lo + 1
                    # 因此两两组合可以加上 m * (m-1) // 2 个，之后就不会有新的答案了，break
                    ans += (hi - lo + 1) * (hi - lo) // 2
                    ans %= MOD
                    break

        return ans % MOD

    def threeSumMulti(self, A: List[int], target: int) -> int:
        # 上述思路中重复的数太多，所以依然超时，
        # 而观察到 A[i] 属于 0 ~ 100 所以我们将其装桶
        MOD = int(1e9 + 7)
        count = collections.Counter(A)
        keys = sorted(count)

        ans = 0
        for i in range(len(keys)):
            T = target - keys[i]
            lo, hi = i, len(keys) - 1  # 注意这里 low 是从 i 开始的，如果 count[keys[i]] > 1 就可以取
            while lo <= hi:  # 这里也变成等于了
                if keys[lo] + keys[hi] < T:
                    lo += 1
                elif keys[lo] + keys[hi] > T:
                    hi -= 1
                else:  # keys[lo] + keys[hi] == T
                    if i < lo < hi:
                        ans += count[keys[i]] * count[keys[lo]] * count[keys[hi]]
                    elif i == lo < hi:
                        ans += count[keys[i]] * (count[keys[i]] - 1) // 2 * count[keys[hi]]
                    elif i < lo == hi:
                        ans += count[keys[i]] * (count[keys[hi]] - 1) * count[keys[hi]] // 2
                    else:  # i==lo ==hi
                        ans += count[keys[i]] * (count[keys[i]] - 1) * (count[keys[i]] - 2) // 6
                    ans %= MOD
                    lo += 1
                    hi -= 1
        return ans % MOD


class Solution_390:
    # math, 找规律
    def lastRemaining(self, n: int) -> int:
        if n == 1:
            return 1
        # 对于 1~n, 我们排除后，就可以得到 2 * [1, 2, ..., n//2]
        # 而重点在于 [1, 2, ..., n//2] 需要重反向开始删除，答案就是
        # 从左边删除的镜像位置，i \in [1, n//2] 的镜像位置 是 （1 + n//2） - i
        return 2 * (1 + n // 2 - self.lastRemaining(n // 2))


class Solution_875:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        def can(K):
            hours = 0
            for p in piles:
                hours += math.ceil(p / K)
            return hours <= H  # (hours + 1 <= H)

        # 解的取值区间[1, max(pile)] 最少吃一个，最多一次吃一堆
        lo = 1
        hi = max(piles)
        while lo < hi:
            mid = (lo + hi) >> 1
            if can(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


class Solution_34:
    def _searchRange(self, nums: List[int], target: int) -> List[int]:
        # O(log n) definitely Binary Search
        lo, hi = 0, len(nums) - 1
        res = [-1, -1]
        while lo <= hi:
            mid = (lo + hi) >> 1
            if nums[mid] == target:
                res[0] = mid
                hi = mid - 1
            elif nums[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1

        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi + 1) >> 1
            if nums[mid] == target:
                res[1] = mid
                lo = mid + 1
            elif nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1

        return res

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binsearch(nums, t, eq):
            lo = 0
            hi = len(nums)  # 可以取到最后一个的后面，这样返回值减一指向最后一个
            while lo < hi:
                mid = (lo + hi) >> 1
                if nums[mid] > target or (eq and target == nums[mid]):
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        left = binsearch(nums, target, eq=True)  # 找第一个大于等于target的
        if len(nums) == left or nums[left] != target:
            return [-1, -1]

        return [left, binsearch(nums, target, eq=False) - 1]  # 第二找大于target的


class Solution_69:
    def mySqrt(self, x: int) -> int:
        # 二分查找
        lo = 0
        hi = x // 2 + 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            t = mid * mid
            if t == x:
                return mid
            elif t < x:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    def mySqrt(self, x: int) -> int:
        # Newton's method
        # f(x) = x^2 - n 求 f(x) = 0的近似根
        # f'(x) = 2*x
        # 牛顿法求0点的迭代式为:
        # x_{k+1} - x_k = f(x_k) / f'(x_k)
        #    x_{k+1} = (x_k ^2 + n)/ 2x_k

        r = x
        while r * r > x:
            r = (r + x // r) // 2
        return r


class Solution_274:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        n = len(citations)
        lo, hi = 0, n
        while lo < hi:
            mi = (lo + hi) >> 1
            if citations[mi] < n - mi:
                lo = mi + 1
            else:
                hi = mi
        return lo

    def hIndex(self, citations):
        # O(n)
        n = len(citations)
        bucket = [0] * (n + 1)
        for x in citations:
            if x >= n:
                bucket[n] += 1
            else:
                bucket[x] += 1
        cnt = 0
        for i in range(n, -1, -1):
            cnt += bucket[i]
            if cnt >= i:
                return i
        return 0


######################################################################################################################

class Solution_12:
    def intToRoman(self, num: int) -> str:
        mp = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX',
              10: 'x_train', 40: 'XL', 50: 'L', 90: 'XC',
              100: 'C', 400: 'CD', 500: 'D', 900: 'CM',
              1000: 'M'}
        res = []
        if num >= 1000:
            res += ['M'] * (num // 1000)
            num %= 1000
        for mul in [100, 10, 1]:
            if num >= 9 * mul:
                res += [mp[9 * mul]]
            elif num >= 5 * mul:
                res += [mp[5 * mul]] + [mp[mul]] * (num // mul - 5)
            elif num >= 4 * mul:
                res += [mp[4 * mul]]
            elif num >= mul:
                res += [mp[mul]] * (num // mul)

            if num >= mul:
                num %= mul
        return ''.join(res)


class Solution_24:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(None)
        dummy.next = head
        q = dummy
        p = t = dummy.next
        while p:
            t = t.next
            if t:
                t = t.next
            else:
                break
            q.next = p.next
            p.next.next = p
            p.next = t
            q = p
            p = t
        return dummy.next


class Solution_29:
    def divide(self, dividend: int, divisor: int) -> int:
        # 二进制除法, 类似于十进制除法，十进制每次乘10，二进制每次乘二（左移一位）
        #            ___101010_
        #       101 / 11010110
        #             101
        #             00110
        #               101
        #               00111
        #                 101
        #                 0100
        sign = (dividend < 0) ^ (divisor < 0)
        a = abs(dividend)
        b = abs(divisor)
        lb, rb = -(1 << 31), (1 << 31) - 1
        res = 0
        while a >= b:
            t = b
            shift = 0
            while a >= t:
                t <<= 1
                shift += 1
            a -= (b << (shift - 1))
            res += 1 << (shift - 1)
        if sign:
            return -res
        return res if res <= rb else rb


class Solution_40:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort(reverse=True)
        res = set()

        def search(i, t, tmp):
            nonlocal res
            if t == target:
                res.add(tuple(tmp))
            if i >= len(candidates) or t > target:
                return
            search(i + 1, t + candidates[i], tmp + [candidates[i]])
            search(i + 1, t, tmp)

        search(0, 0, [])
        return list(res)


#######################################################################################################################
# 一维度区间问题
# 两个一维区间(a, b), (c, d)相交的情况是
#    a----c---b----d
#    c----a---d----b
#    a----c----d---b
#    c----a----b---d
# 于是相交的条件是 a <= d and c <= b (是否取等号视具体情况)
#
# 一维区间的交 [1, 3] ∩ [2, 4] = [2, 3]
# 相交的两个一维区间的并 [1, 3] U [2, 4] = [1, 4]

class Solution_56:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        def intersect(in1, in2):
            a, b = in1
            c, d = in2
            return a <= d and c <= b

        if len(intervals) < 2:
            return intervals
        inters = sorted(intervals)
        i = 0
        while i < len(inters) - 1:
            if intersect(inters[i], inters[i + 1]):
                # since inters is ascending order inters[i][0] <= inters[i+1][0], we only need to merge right bound
                inters[i][1] = max(inters[i][1], inters[i + 1][1])
                inters.pop(i + 1)
            else:
                i += 1
        return inters

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []
        inter = sorted(intervals)
        res = []
        # print(inter)
        x, y = inter[0]
        for a, b in inter[1:]:
            if x <= b and a <= y:
                x = min(x, a)
                y = max(y, b)
            else:
                res.append([x, y])
                x, y = a, b
        if not res or res[-1] != [x, y]:
            res.append([x, y])
        return res

    def merge(self, intervals):
        inters = sorted(intervals)
        # since we have sorted, we only need last interval's end < current to ensure no overlap
        merged = []
        for inter in inters:
            if not merged or merged[-1][1] < inter[0]:
                merged.append(inter)
            else:
                merged[-1][1] = max(merged[-1][1], inter[1])
        return merged

    def merge(self, intervals):
        # 扫描线算法
        # [[1,3],[2,6],[8,10],[15,18]]
        # 1( 2( 3) 6) 8( 10) 15( 18)
        # 只看括号就是 (()) () ()
        Left, Right = 0, 1  # 先处理 Left

        events = []
        for inter in intervals:
            events.append((inter[0], Left))
            events.append((inter[1], Right))
        events.sort()

        res = []
        balance = 0
        prev = -1
        for x, T in events:
            if balance == 0:  # balance 表示左右()是否平衡了, 类似括号匹配了
                prev = x
            balance += (1 if T == Left else -1)
            if balance == 0:  # 处理之后平衡了就新加一个结果
                res.append([prev, x])
        return res


class Solution_57:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # 二分查找，插入，然后 使用56 题中的 merge
        bisect.insort(intervals, newInterval)
        res = []
        for inter in intervals:
            if not res or res[-1][1] < inter[0]:
                res.append(inter)
            else:
                res[-1][1] = max(res[-1][1], inter[1])
        return res


class Solution_759:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        Left, Right = 0, 1
        events = []
        for a, b in intervals:
            events.append([a, Left])
            events.append([b, Right])

        events.sort()

        res = []
        balance = 0
        prev = 0
        for x, T in events:
            if balance == 0:
                res.append([prev, x])
            balance += (1 if T == Left else -1)
            if balance == 0:
                prev = x
        return res[1:]


class Solution_495:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        res = 0
        last = 0
        for st in timeSeries:
            if st < last:
                res += duration - (last - st)
            else:
                res += duration
            last = st + duration
        return res

    def _findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        res = 0
        if len(timeSeries) == 0:
            return 0
        for i in range(len(timeSeries) - 1):
            res += min(timeSeries[i + 1] - timeSeries[i], duration)
        return res + duration


class RangeModule_715:
    # TODO 这一题还没写完
    def __init__(self):
        self.ranges = []

    def addRange(self, left: int, right: int) -> None:
        if not self.ranges:
            self.ranges.append([left, right])
        else:
            if right < self.ranges[0][0]:
                self.ranges.insert(0, [left, right])
                return
            if left > self.ranges[-1][1]:
                self.ranges.append([left, right])
                return
            i = 0
            while i < len(self.ranges):
                if self.intersect(self.ranges[i], [left, right]):
                    self.ranges[i] = [min(self.ranges[i][0], left), max(self.ranges[i][1], right)]
                    j = i
                    while j < len(self.ranges) - 1:
                        if self.intersect(self.ranges[j], self.ranges[j + 1]):
                            self.ranges[i] = [min(self.ranges[j][0], self.ranges[j + 1][0]),
                                              max(self.ranges[j][1], self.ranges[j + 1][1])]
                            self.ranges.pop(j + 1)
                        else:
                            j += 1
                    return
                elif i < len(self.ranges) - 1 and self.ranges[i][1] < left and right < self.ranges[i + 1][0]:
                    self.ranges.insert(i + 1, [left, right])
                    return

    def queryRange(self, left: int, right: int) -> bool:
        pass

    def removeRange(self, left: int, right: int) -> None:
        pass

    @staticmethod
    def intersect(in1, in2):
        a, b = in1
        c, d = in2
        return a <= d and c <= b


class Solution_763:
    def partitionLabels(self, S: str) -> List[int]:
        # Greedy
        last = {}
        for i, c in enumerate(S):
            last[c] = i
        # print(last)
        res = []
        i = 0
        while i < len(S):
            end = last[S[i]]
            j = i + 1
            while j < end:
                if last[S[j]] > end:
                    end = last[S[j]]
                j += 1
            res.append(end - i + 1)
            i = end + 1

        return res


class Solution_986:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        def intersect(in1, in2):
            a, b = in1
            c, d = in2
            if a <= d and c <= b:
                return [max(a, c), min(b, d)]
            else:
                return []

        i = j = 0
        merged = []
        while i < len(A) and j < len(B):
            interval = intersect(A[i], B[j])
            if interval:
                merged.append(interval)
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return merged

    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i = j = 0
        merged = []
        while i < len(A) and j < len(B):
            a = max(A[i][0], B[j][0])
            b = min(A[i][1], B[j][1])
            if a <= b:
                merged.append([a, b])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1

        return merged


#######################################################################################################################


class Solution_725:
    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
        n = 0
        p = root
        while p:
            n += 1
            p = p.next

        cnt, r = divmod(n, k)
        res = []
        p = root
        for i in range(k):
            if p:
                num = cnt + (1 if len(res) < r else 0)
                res.append(p)
                tmp = 0
                t = p
                while tmp < num:
                    t = p
                    p = p.next
                    tmp += 1
                t.next = None
            else:
                res.append(None)

        return res


class Solution_60:
    def _getPermutation(self, n: int, k: int) -> str:
        # brute force accept
        it = itertools.permutations(list(range(1, n + 1)))
        res = []
        while k > 0:
            res = next(it)
            k -= 1
        return ''.join([str(x) for x in res])

    def getPermutation(self, n: int, k: int) -> str:
        memo = {1: 1}
        for i in range(2, 10):
            memo[i] = memo[i - 1] * i

        # print(memo)

        def foo(n, k, ls):
            # print(f'n={n:>}, k={k:>}, ls={ls}')
            if k <= 1:
                return ls
            elif n <= 2:
                return ls[::-1]
            idx, kt = divmod(k, memo[n - 1])
            if kt == 0:
                return [ls[idx - 1]] + ls[idx:][::-1] + ls[:idx - 1][::-1]
            tmp = ls[:idx] + ls[idx + 1:]
            return [ls[idx]] + foo(n - 1, kt, tmp)

        res = foo(n, k, list(range(1, n + 1)))
        return ''.join([str(x) for x in res])


class Solution_75:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # Two pointers
        i = j = 0
        k = len(nums) - 1

        while i <= j <= k:
            if nums[j] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j += 1
            elif nums[j] == 2:
                nums[j], nums[k] = nums[k], nums[j]
                k -= 1
            else:
                j += 1
            # print((i, j, k), nums)


class Solution_86:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 链表划分，这个可以用到链表的快排中？
        tmp_list = ListNode(None)
        tmp_tail = tmp_list

        dummy = ListNode(None)
        dummy.next = head
        q = dummy
        p = head
        while p:
            if p.val >= x:
                q.next = p.next
                p.next = None
                tmp_tail.next = p
                tmp_tail = tmp_tail.next
                p = q.next
            else:
                q = p
                p = p.next

        q.next = tmp_list.next
        return dummy.next


class Solution_148:
    # TODO
    def _sortList(self, head: ListNode) -> ListNode:

        def quick_sort(head):
            if not head or not head.next:
                return head
            p = pivot = head
            smaller = ListNode(None)
            greater = ListNode(None)
            stail = smaller
            gtail = greater
            t = head.next
            while t:
                if t.val < pivot.val:
                    stail.next = t
                    stail = stail.next
                elif t.val == pivot.val:
                    p.next = t
                    p = p.next
                else:
                    gtail.next = t
                    gtail = gtail.next
                t = t.next
            stail.next = None
            gtail.next = None
            p.next = None

            smaller.next = quick_sort(smaller.next)
            greater.next = quick_sort(greater.next)

            if smaller.next:
                t = smaller.next
                while t.next:
                    t = t.next
                t.next = pivot
                p.next = greater.next
                return smaller.next
            else:
                p.next = greater.next
                return pivot

        return quick_sort(head)

    def sortList(self, head):
        if not head or not head.next:
            return head

        p = pivot = head
        l1 = ListNode(0)
        l2 = ListNode(0)
        s = l1
        f = l2
        tmp = head.next
        while tmp is not None:
            if tmp.val < pivot.val:
                s.next = tmp
                s = s.next
            elif tmp.val == pivot.val:
                p.next = tmp
                p = p.next
            else:
                f.next = tmp
                f = f.next
            tmp = tmp.next
        s.next = None
        f.next = None
        p.next = None
        l3 = self.sortList(l1.next)
        l4 = self.sortList(l2.next)
        if l3 is not None:
            l5 = l3
            while l5.next is not None:
                l5 = l5.next
            l5.next = pivot
            p.next = l4
            return l3
        else:
            p.next = l4
        return pivot


head = ListNode(None)
tail = head
for x in [4, 3, 2, 1]:
    tail.next = ListNode(x)
    tail = tail.next

print(Solution_148()._sortList(head.next))


class Solution_89:
    def grayCode(self, n: int) -> List[int]:
        res = []
        s = set()

        def foo(k):
            nonlocal res, s
            s.add(k)
            res.append(k)
            for i in range(n):
                t = k ^ (1 << i)
                if t not in s:
                    foo(t)

        foo(0)
        return res

    def grayCode(self, n: int) -> List[int]:
        # 这个是参考别人的做法， 这个递归写的很好，深刻理解了生成gray code的含义
        def changeOne(n):
            if n == 0:
                return [0]
            elif n == 1:
                return [0, 1]
            else:
                subset = changeOne(n - 1)
                tmp = [x | (1 << (n - 1)) for x in subset]
                return subset + tmp[::-1]

        return changeOne(n)


class Solution_96:
    # 卡特兰数的算法
    def numTrees(self, n: int) -> int:
        # def catalan(n):
        #     if n <= 2:
        #         return n
        #     return ( (4*n + 2) // (n + 2) ) * catalan(n-1)
        res = [1] * (n + 1)
        for i in range(n):
            res[i + 1] = ((4 * i + 2) * res[i]) // (i + 2)
        return res[n]

    def numTrees(self, n: int) -> int:
        def catalan(n):
            if n <= 2:
                return n
            return ((4 * n - 2) * catalan(n - 1)) // (n + 1)

        return catalan(n)


#######################################################################################################################
# 位操作
# |, &, ^, ~, >>, << (>>>)
# 一些骚操作：
#   x & 1 判断x奇偶性
#   x & (x - 1) 将最x右面一位1置零(可以用来判断x是不是2的幂， 数x中为1的位数)
#   x ^ (x & (x-1)) 返回x最右边一位为1的数
#   x & (-x) 返回x最右边一位1的数
#   r & ~(r - 1) 求r的最右边一位1
#   a^=b; b^=a; a^=b  交换a,b两个值
#   (a ^ (a>>31)) - (a>>31) 对a取绝对值，python不可以这么干

class Solution_371:
    def getSum(self, a, b):
        # Bit manipulation
        # 使用二进制循环进位加法器的原理
        mask = 0xffffffff
        res = 0
        c = 0
        for i in range(32):
            ai = ((a & mask) >> i) & 1
            bi = ((b & mask) >> i) & 1
            s = (ai ^ bi ^ c)
            c = (ai & bi) | (ai & c) | (bi & c)
            res |= (s << i)
        # 为了处理负数的问题
        bins = bin(res & mask)[2:].zfill(32)
        return int(bins[1:], 2) - int(bins[0]) * (1 << 31)

    def _getSum(self, a: int, b: int) -> int:
        # Bit manipulation
        # TODO 看不懂这个
        if a == 0 or b == 0:
            return a | b
        mask = 0xffffffff
        while b & mask:
            c = (a & b) << 1
            a = a ^ b
            b = c

        return a & mask if b > mask else a


class Solution_136:
    def singleNumber(self, nums: List[int]) -> int:
        return functools.reduce(lambda x, y: x^y, nums)


class Solution_137:
    def singleNumber(self, nums: List[int]) -> int:
        # bit manipulation
        # 对于每一位如果 1 出现的总次数是3的倍数，则该位为0，否则该位为1
        res = 0
        for i in range(32):
            v = 1 << i
            s = sum([x & v != 0 for x in nums])
            if s % 3 != 0:
                res |= v
        if res > (1 << 31) - 1:
            return (res & 0x7fffffff) - (1 << 31)  # python负数
        return res

    def singleNumber(self, nums: List[int]) -> int:
        # using extra space for set
        return (3 * (sum(set(nums))) - sum(nums)) // 2

    def singleNumber(self, nums):
        # 第一次出现，seenTwice^x == x, ~seenTwice == -1，然后seenOnce被赋值为x
        #            seenTwice^x == x, x & (~seenOnce) == x & (~x) ==  0, seenTwice被赋值为0
        # 第二次出现，seenOnce ^ x == 0, ~seenTwice == -1, seenOnce被赋值为0
        #            seenTwice^x  == x, ~seenOnce == -1，seenTwice赋值为x
        # 第三次出现，seenOnce ^ x == x, x & ~seenTwice == x & (~x) ==  0,seenOnce被赋值为0
        #            seenTwice^x  == 0, ~seenOnce = -1, seenTwice被赋值为0
        seenOnce = seenTwice = 0
        for x in nums:
            seenOnce = (seenOnce ^ x) & ~seenTwice
            seenTwice = (seenTwice ^ x) & ~seenOnce
        return seenOnce


class Solution_260:
    def singleNumber(self, nums: List[int]) -> List[int]:
        r = 0
        for n in nums:
            r ^= n
        # 上述循环结束后 r = a^b (a, b是要找的两个只出现一次的数)，
        # 其最低位的1一定是由于a,b对应位置不同才出现的 0^1 = 1^0 = 1,
        # 于是通过判断nums中数的这一位为0还是为1可以将nums分为两组，
        # 一组是包含a,和其他出现两次的数；另一组是包含b和剩下出现两次的数。
        # 这样分别对两组数异或，就可以求出a, b了
        mask = r & ~(r - 1)
        # 下面注释代码块和上述语句实现同样的功能
        # mask = 1
        # for i in range(32):
        #     if r & mask == mask:
        #         break
        #     mask<<=1

        res = [0, 0]
        for n in nums:
            if n & mask:
                res[0] ^= n
            else:
                res[1] ^= n

        return res


class Solution_78:
    def _subsets(self, nums: List[int]) -> List[List[int]]:
        res = []

        def foo(tmp, nums, i):
            nonlocal res
            if i < len(nums):
                foo(tmp + [nums[i]], nums, i + 1)
                foo(tmp, nums, i + 1)
            else:
                res += [tmp]

        foo([], nums, 0)
        return res

    def _subsets(self, nums):
        res = [[]]
        for x in nums:
            tmp = []
            for r in res:
                t = r[:]
                t.append(x)
                tmp.append(t)
            res += tmp
        return res

    def subsets(self, nums):
        # 使用位操作来生成所有子集（注意溢出问题）
        # 对应位i为 1 表示取nums[i]这个元素，否则不取
        t = 1 << len(nums)
        res = []
        for i in range(t):
            tmp = []
            for j in range(len(nums)):
                if i & (1 << j):
                    tmp.append(nums[j])
            res.append(tmp)
        return res


class Solution_1177:
    def canMakePaliQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        # substring t can be rearranged, so check whether
        # the number of letter that appear odd times <= 1 + 2*k
        # 高效表示perfix character frequency应该使用bitmap
        # 奇偶直接用0-1表示，加换成异或

        def number1bit(x):
            cnt = 0
            while x:
                x = x & (x - 1)
                cnt += 1
            return cnt

        count = [0 for _ in range(len(s))]
        for i, c in enumerate(s):
            if i > 0:
                count[i] |= count[i - 1]
            idx = ord(c) - ord('a')
            count[i] ^= (1 << idx)

        ans = []
        for query in queries:
            left, right, k = query
            if k >= min((right - left + 1), 26) // 2:
                ans.append(True)
                continue
            cnt = count[right] ^ (count[left - 1] if left > 0 else 0)
            s = number1bit(cnt)
            # print(memo)
            ans.append(s - 2 * k <= 1)
        return ans


class Solution_1007:
    def minDominoRotations(self, A: List[int], B: List[int]) -> int:
        # 用对应二进制位表示状态
        t = -1
        eq = 0
        for a, b in zip(A, B):
            eq += (a == b)
            tmp = (1 << a) | (1 << b)
            t &= tmp

        c = 0
        for i in range(1, 7):
            if (t >> i) & 1:
                c = i

        if c == 0:
            return -1

        swaps = 0
        for a in A:
            if c != a:
                swaps += 1

        return min(swaps, len(A) - eq - swaps)


class Solution_338:
    def _countBits(self, num: int) -> List[int]:
        res = []

        def count(x):
            cnt = 0
            while x:
                x &= (x - 1)
                cnt += 1
            return cnt

        for i in range(num + 1):
            res.append(count(i))
        return res

    def countBits(self, num: int) -> List[int]:
        def gen(n):
            if n == 2:
                return [0, 1]
            t = gen(n // 2)
            return t + [x + 1 for x in t]

        def nextPow2(n):
            # n -= 1
            n |= n >> 16
            n |= n >> 8
            n |= n >> 4
            n |= n >> 2
            n |= n >> 1
            return n + 1

        if num < 2:
            t = 2
        else:
            t = nextPow2(num)
        ls = gen(t)
        return ls[:num + 1]


#######################################################################################################################

class Solution_11:
    def maxArea(self, height: List[int]) -> int:
        # two pointers
        i, j = 0, len(height) - 1
        m = 0
        ml, mr = height[i], height[j]
        while i < j:
            m = max(m, (j - i) * min(height[i], height[j]))
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1
        return m


class Solution_42:
    def trap(self, height: List[int]) -> int:
        # 对每个 bin 利用左右高点的方式来算
        # 先存一遍右边的最高点
        right = [0] * (len(height) + 1)
        for i in range(len(height) - 1, -1, -1):
            right[i] = max(right[i + 1], height[i])
        right = right[1:]

        left_m = 0
        res = 0
        for i in range(len(height)):
            t = min(left_m, right[i]) - height[i]
            if t > 0:
                res += t
            # 左边的最高点
            left_m = max(left_m, height[i])
        return res

    def trap(self, height: List[int]) -> int:
        # two pointers, O(n) time, O(1) space
        # 不使用额外的数组空间维护左/右侧的最大值，而是使用双指针法来做
        lo, hi = 0, len(height) - 1
        res = 0
        lm = rm = 0  # 左侧最大值和右侧的最大值
        while lo < hi:
            # height[lo] <= height[hi] 表示右边当前值高于或等于左边，这样我们如果左边 lm > height[lo] 的话就可能积水
            if height[lo] <= height[hi]:
                lm = max(lm, height[lo])
                res += lm - height[lo]  # lm - height[lo] >= 0
                lo += 1
            # 同理，height[lo] > height[hi] 表示左边当前值高于右边，这样我们如果右边 rm > height[hi] 的话也会积水
            else:
                rm = max(rm, height[hi])
                res += rm - height[hi]
                hi -= 1
        return res

    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0

        for i, h in enumerate(height):
            # 使用 stack 维护降序序列，这个对于一块连着的积水时一层一层算的。
            while stack and height[stack[-1]] < h:
                t = stack.pop()
                if stack:
                    min_h = min(height[stack[-1]], h)
                    res += (min_h - height[t]) * (i - stack[-1] - 1)
            stack.append(i)
        return res


class Solution_152:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        # 维护正的最大值，和负的最小值（负的最小值乘上一个数有可能成为最大值）
        dp = nums[0]
        left = min(nums[0], 0)
        m = max(dp, 0)
        for i, x in enumerate(nums[1:]):
            tmp = max(x * dp, x, x * left)
            if m < tmp:
                m = tmp
            left = min(x * dp, x, x * left, 0)
            dp = tmp
        return m


class Solution_238:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 先从右向左存一遍右边的积，再从左向右存乘上左边的积
        r = 1
        right = [1] * len(nums)
        for j in range(len(nums) - 2, -1, -1):
            right[j] = nums[j + 1] * right[j + 1]
        for i in range(len(nums)):
            right[i] = r * right[i]
            r *= nums[i]
        return right


class Solution_407:

    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        for ls in heightMap:
            print(ls)
        # TODO 看不懂
        if len(heightMap) < 3:
            return 0
        m, n = len(heightMap), len(heightMap[0])
        if n < 3:
            return 0
        unit_hold = [[20001] * n for _ in range(m)]
        que = []
        for i in range(m):
            unit_hold[i][0] = heightMap[i][0]
            unit_hold[i][-1] = heightMap[i][-1]
            que.append((i, 0))
            que.append((i, n - 1))
        for j in range(n):
            unit_hold[0][j] = heightMap[0][j]
            unit_hold[-1][j] = heightMap[-1][j]
            que.append((0, j))
            que.append((m - 1, j))

        while que:
            x, y = que.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    bound = max(heightMap[nx][ny], unit_hold[x][y])
                    if unit_hold[nx][ny] > bound:
                        unit_hold[nx][ny] = bound
                        que.append((nx, ny))

        res = 0
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                res += unit_hold[i][j] - heightMap[i][j]
        return res

    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        # TODO 从最外层开始，一层一层向内围，通过max(h, heightMap[nx][ny]) 来保持目前最高的bound
        m = len(heightMap)
        n = len(heightMap[0]) if m else 0
        if m < 3 or n < 3:
            return 0

        visit = set()
        que = []
        for j in [0, n - 1]:
            for i in range(m):
                que.append((heightMap[i][j], i, j))
                visit.add((i, j))
        for i in [0, m - 1]:
            for j in range(n):
                que.append((heightMap[i][j], i, j))
                visit.add((i, j))
        heapq.heapify(que)  # 建堆
        res = 0
        while que:
            h, x, y = heapq.heappop(que)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visit:
                    res += max(0, h - heightMap[nx][ny])
                    heapq.heappush(que, (max(h, heightMap[nx][ny]), nx, ny))
                    visit.add((nx, ny))

        return res


class Solution_37:
    def _solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        n = len(board)
        row = [[0] * n for _ in range(n)]
        col = [[0] * n for _ in range(n)]
        box = [[0] * n for _ in range(n)]

        def foo(board, x, y):
            if y == n:  # 第x行处理完了
                x += 1
                y = 0
            if x == n:  # board处理完了
                return True
            if board[x][y] != '.':
                return foo(board, x, y + 1)  # 这个不用填，下一个
            for c in range(1, n + 1):
                if row[x][c - 1] or col[y][c - 1] or box[x // 3 * 3 + y // 3][c - 1]:
                    continue  # 行/列或者块中填过这数字了
                board[x][y] = str(c)
                row[x][c - 1] = col[y][c - 1] = box[x // 3 * 3 + y // 3][c - 1] = 1
                if foo(board, x, y + 1):
                    return True
                row[x][c - 1] = col[y][c - 1] = box[x // 3 * 3 + y // 3][c - 1] = 0
                board[x][y] = '.'
            return False

        for i in range(n):
            for j in range(n):
                c = board[i][j]
                if c != '.':
                    row[i][int(c) - 1] = col[j][int(c) - 1] = box[i // 3 * 3 + j // 3][int(c) - 1] = 1

        foo(board, 0, 0)

    def solveSudoku(self, board: List[List[str]]) -> None:
        n = len(board)
        row = [[0] * n for _ in range(n)]
        col = [[0] * n for _ in range(n)]
        box = [[0] * n for _ in range(n)]
        to_fill = []
        for i in range(n):
            for j in range(n):
                c = board[i][j]
                if c != '.':
                    row[i][int(c) - 1] = col[j][int(c) - 1] = box[i // 3 * 3 + j // 3][int(c) - 1] = 1
                else:
                    to_fill.append((i, j))

        def get_candidate(x, y):
            for c in range(1, n + 1):
                if row[x][c - 1] == 0 and col[y][c - 1] == 0 and box[x // 3 * 3 + y // 3][c - 1] == 0:
                    yield c

        def foo(idx, board):
            if idx == len(to_fill):
                return True
            x, y = to_fill[idx]
            for c in get_candidate(x, y):
                board[x][y] = str(c)
                row[x][c - 1] = col[y][c - 1] = box[x // 3 * 3 + y // 3][c - 1] = 1
                if foo(idx + 1, board):
                    return True
                row[x][c - 1] = col[y][c - 1] = box[x // 3 * 3 + y // 3][c - 1] = 0
                board[x][y] = '.'
            return False

        foo(0, board)


class Solution_51:
    def solveNQueens(self, n: int) -> List[List[str]]:
        from copy import deepcopy
        row = [0] * n
        col = [0] * n
        board = [['.'] * n for _ in range(n)]
        res = []
        cnt = 0

        def can(board, x, y):
            if row[x] == 1 or col[y] == 1:
                return False
            i, j = x, y
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            i, j = x, y
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        def foo(board, idx):
            nonlocal res, cnt
            if idx == n:
                cnt += 1
                res.append([''.join(ls) for ls in board])
                return
            for j in range(n):
                if can(board, idx, j):
                    board[idx][j] = 'Q'
                    row[idx] = col[j] = 1
                    foo(board, idx + 1)
                    row[idx] = col[j] = 0
                    board[idx][j] = '.'

        foo(board, 0)
        return res  # return cnt  solve 52


class Solution_130:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        import collections
        m = len(board)
        n = len(board[0]) if m else 0
        if m == 0 or n == 0:
            return
        visit = [[0 if board[i][j] == 'O' else 1 for j in range(n)] for i in range(m)]

        def dfs(i, j):
            nonlocal visit
            visit[i][j] = 1
            que = collections.deque([(i, j)])
            to_use = [(i, j)]
            res = True
            while que:
                x, y = que.popleft()
                if x == 0 or x == m - 1 or y == 0 or y == n - 1:
                    res = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and visit[nx][ny] == 0:
                        to_use.append((nx, ny))
                        que.append((nx, ny))
                        visit[nx][ny] = 1
            return to_use, res

        to_use = []
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if visit[i][j] == 0:
                    to, b = dfs(i, j)
                    if b:
                        to_use.extend(to)
        for x, y in to_use:
            board[x][y] = 'x_train'

        for ls in board:
            print(ls)


class Solution_53:
    def _maxSubArray(self, nums: List[int]) -> int:
        m = nums[0]
        b = nums[0]
        for i in range(1, len(nums)):
            # 全是非负数的话，整个数组一定是最大的
            # 如果前面部分大于0的话，加上当前数有可能成为最大值
            # 当前数加上一个负数不可能等于最大值
            b = b + nums[i] if b > 0 else nums[i]
            m = max(m, b)
        return m

    def maxSubArray(self, nums):
        # DP
        m = nums[0]
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i] + nums[i - 1])
            m = max(m, nums[i])
        return m


class Solution_121:
    def _maxProfit(self, prices: List[int]) -> int:
        diff = [0]
        for i in range(len(prices) - 1):
            diff.append(prices[i + 1] - prices[i])
        # convert to [53. Maximum Subarray]
        m = diff[0]
        for i in range(1, len(diff)):
            diff[i] = max(diff[i], diff[i] + diff[i - 1])
            m = max(m, diff[i])

        return m

    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        min_price = float('inf')
        for i in range(len(prices)):
            max_profit = max(max_profit, prices[i] - min_price)
            min_price = min(min_price, prices[i])
        return max_profit


class Solution_122:
    def maxProfit(self, prices: List[int]) -> int:
        # 因为可以任意多个transaction, 所以有利润就加上去
        profit = 0
        for i in range(len(prices) - 1):
            t = prices[i + 1] - prices[i]
            if t > 0:
                profit += t
        return profit


class Solution_123:
    def maxProfit(self, prices: List[int]) -> int:
        # O(k·n^2) TLE
        # dp[k, i] = max(dp[k, i-1], prices[i] - prices[j] + dp[k-1, j-1]) 其中 j ~ [0, ..., i-1]
        # 当取 dp[k, i-1] 时，到这一步，不做交易
        # 当取 prices[i] - prices[j] + dp[k-1, j-1] 表示做一笔交易 （j 买入， i 卖出），然后加上少一次交易(k-1, j-1)的收益
        # 最后结果时 dp(2, n-1)
        if not prices:
            return 0
        n = len(prices)
        dp = [[0] * n for _ in range(3)]
        for k in range(1, 3):
            for i in range(1, n):
                mini = prices[0]
                for j in range(1, i + 1):
                    mini = min(mini, prices[j] - dp[k - 1][j - 1])
                dp[k][i] = max(dp[k][i - 1], prices[i] - mini)
        return dp[2][n - 1]

    def maxProfit(self, prices: List[int]) -> int:
        # O(k·n)
        # 时间优化，上面的解法是 O(k·n^2) ，实际上我们不用每次 j 从零枚举到 i-1, 我们一边遍历一遍保存下最小值就好了
        # 这样时间复杂度为 O(k·n).
        # 另外，由于只依赖于 k-1 行，空间也可以优化为 O(n)， 但由于 k == 2, 所以影响不是很大~
        if not prices: return 0
        n = len(prices)
        dp = [[0] * n for _ in range(2)]
        for k in range(2):
            mini = prices[0]
            for i in range(1, n):
                mini = min(mini, prices[i] - (dp[k - 1][i - 1] if k > 0 else 0))
                dp[k][i] = max(dp[k][i - 1], prices[i] - mini)
        # print(dp)
        return dp[1][n - 1]

    def maxProfit(self, prices: List[int]) -> int:
        # 这个就是一个比较神奇的做法，O(n) time, O(1) space
        if not prices: return 0
        one_buy = two_buy = prices[0]
        profit_one = profit_two = 0
        for p in prices:
            one_buy = min(one_buy, p)  # lowest price
            profit_one = max(profit_one, p - one_buy)  # max price - lowest price
            two_buy = min(two_buy, p - profit_one)  # use profit_one to buy another
            profit_two = max(profit_two, p - two_buy)
        return profit_two


class Solution_188:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        # 这个就是  Best Time to Buy and Sell Stock III 的泛化
        # MLE, k 有可能非常大，因此需要使用空间优化, 然而空间优化后， k 非常大还是会 TLE
        if not prices or k <= 0: return 0  # don't forget edge case
        n = len(prices)
        dp = [[0] * n for _ in range(k)]

        for kk in range(k):
            m = prices[0]
            for i in range(1, n):
                m = min(m, prices[i] - (dp[kk - 1][i - 1] if kk > 0 else 0))
                dp[kk][i] = max(dp[kk][i - 1], prices[i] - m)
        return dp[k - 1][n - 1]

    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices or k <= 0: return 0  # don't forget edge case
        n = len(prices)
        if k >= n // 2:  # k 超过 n 一半时，最多可以做 n//2 个交易
            res = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    res += prices[i] - prices[i - 1]
            return res

        dp = [0] * n
        for kk in range(k):
            m = prices[0]
            dp2 = [0] * n
            for i in range(1, n):
                m = min(m, prices[i] - dp[i - 1])
                dp2[i] = max(dp2[i - 1], prices[i] - m)
            dp[:] = dp2[:]
        return dp[n - 1]

    def maxProfit(self, k: int, prices: List[int]) -> int:
        # using max_heap and stack
        # 1. 找出所有的极小值点-极大值点对 (valley, peek) -> (v, p)
        # 2. 使用栈将这些点对处理成可能的 profit，存入 max_heap
        # 3. 取 max_heap 前 k 大的和为结果
        #
        n = len(prices)
        res = v = p = 0
        pool, stack = [], []

        while p < n:
            # 找一对 valley, peek
            v = p
            while v < n - 1 and prices[v] >= prices[v + 1]:
                v += 1

            p = v + 1
            if p >= n: break
            while p < n - 1 and prices[p] <= prices[p + 1]:
                p += 1

            # 新找到一次交易买入价格更低，出栈，把出栈的可能 profit 放入堆
            while stack and prices[v] <= prices[stack[-1][0]]:
                tv, tp = stack.pop()
                pool.append(-(prices[tp] - prices[tv]))

            # 这里是这个算法最为 trick 的地方
            # 处理的是两对交易能否合并， 假如两次交易 (v1, p1), (v2, p2) 有 p2 > p1 > v2
            # 如果只能再进行一次交易: 那么就是 prices[p2] - prices[v1]  # 最高减最低
            # 如果还能再进行两次交易： 那么就是 (prices[p2] - prices[v2]) + (prices[p1] - prices[v1])
            # 对后面一个式子变形一下，写成 (prices[p2] - prices[v1]) + (prices[p1] - prices[v2])
            # 前面一项就是只能再进行一次交易时的利润。如果可以进行两次，我们还可以另外
            # 获得 (prices[p1] - prices[v2]) 这部分利润（类似于股票做 T）
            # 上一个循环跳出时有 prices[v] > prices[stack[-1][0]] 即 v2 > v1
            # 下面这个循环判断的是 prices[p] >= prices[stack[-1][1]] 即 p2 > p1
            # 注意不会出现 p2 > v2 > p1 > v1 这种情况，这样 p1, v2 不是极值点
            while stack and prices[p] >= prices[stack[-1][1]]:
                tv, tp = stack.pop()
                pool.append(-(prices[tp] - prices[v]))
                v = tv
            stack.append([v, p])

        while stack:
            tv, tp = stack.pop()
            pool.append(-(prices[tp] - prices[tv]))

        heapq.heapify(pool)
        i = 0
        while i < k and pool:
            res -= heapq.heappop(pool)
            i += 1
        return res


class Solution_335:
    def isSelfCrossing(self, x: List[int]) -> bool:
        # 3 case

        if len(x) < 4:
            return False
        for i in range(3, len(x)):
            if x[i] >= x[i - 2] and x[i - 1] <= x[i - 3]:
                return True
            if i >= 4 and x[i - 1] == x[i - 3] and x[i - 2] <= x[i] + x[i - 4]:
                return True
            if i >= 5 and x[i - 3] >= x[i - 1] >= x[i - 3] - x[i - 5] and x[i] >= x[i - 2] - x[i - 4] and x[i - 4] <= x[
                i - 2]:
                return True
        return False


class Solution_30:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        #
        if not s or not words:  # 边界情况，s / word 为空
            return []

        m = len(words[0])
        n = len(words)
        words = sorted(words)
        ws = set(words)
        t = m * n
        res = []
        for i in range(len(s) - t + 1):
            if s[i:i + m] in ws:
                tmp = [s[j:j + m] for j in range(i, i + t, m)]
                if sorted(tmp) == words:
                    res.append(i)
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


######################################################################################################################
# 左右两边扫描

class Solution_838:
    def pushDominoes(self, dominoes: str) -> str:
        # 两遍的解法， 一遍从左向右，一遍从右向左
        # 和 821 差不多
        N = len(dominoes)
        force = [0] * N

        f = 0
        for i in range(N):
            if dominoes[i] == 'R':
                f = N
            elif dominoes[i] == 'L':
                f = 0
            else:
                f = max(f - 1, 0)
            force[i] += f

        f = 0
        for i in range(N - 1, -1, -1):
            if dominoes[i] == 'L':
                f = N
            elif dominoes[i] == 'R':
                f = 0
            else:
                f = max(f - 1, 0)
            force[i] -= f

        tmp = []
        for f in force:
            if f == 0:
                tmp.append('.')
            elif f > 0:
                tmp.append('R')
            else:
                tmp.append('L')
        return ''.join(tmp)


class Solution_845:
    def longestMountain(self, A: List[int]) -> int:
        # O(n) time, O(1) space
        # 我们使用两次遍历，从左到右，从右到左
        n = len(A)
        up, down = [0] * n, [0] * n
        for i in range(1, n):
            if A[i] > A[i - 1]:
                up[i] = up[i - 1] + 1

        for i in range(n - 2, -1, -1):
            if A[i] > A[i + 1]:
                down[i] = down[i + 1] + 1

        res = 0
        # 上下必须都大于零的才算
        for i in range(n):
            if up[i] > 0 and down[i] > 0:
                res = max(res, up[i] + down[i] + 1)

        return res

    def longestMountain(self, A: List[int]) -> int:
        res = up = down = 0
        for i in range(1, len(A)):
            # 前面是下降 且变上升 -> 谷底
            # 前一个和当前相等了
            if (down and A[i - 1] < A[i]) or A[i - 1] == A[i]:
                up = down = 0
            up += A[i - 1] < A[i]  # 上升
            down += A[i - 1] > A[i]  # 下降
            if up and down:
                res = max(res, up + down + 1)
        return res


class Median:
    def __init__(self):
        self.max_heap = []  # smaller half
        self.min_heap = []  # bigger half

    def insert(self, x):
        heapq.heappush(self.max_heap, -heapq.heappushpop(self.min_heap, x))

        if len(self.max_heap) > len(self.min_heap):
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

    def median(self):
        if len(self.min_heap) == len(self.max_heap):
            return (self.min_heap[0] - self.max_heap[0]) / 2
        return self.min_heap[0]

    def remove(self, x):
        def remove_from_heap(pool, target):
            index = pool.index(target)
            pool[index] = pool[-1]
            pool.pop()
            if index < len(pool):
                heapq.heapify(pool)

        if self.min_heap[0] <= x:
            remove_from_heap(self.min_heap, x)
            if len(self.max_heap) > len(self.min_heap):
                heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        else:
            remove_from_heap(self.max_heap, -x)
            if len(self.min_heap) > len(self.max_heap) + 1:
                heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def __repr__(self):
        return f'{self.max_heap},{self.min_heap}'


class Solution_480:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        if not nums:
            return []

        m = Median()

        for i in range(k):
            m.insert(nums[i])
            print(m)

        res = [m.median()]
        for i in range(k, len(nums)):
            m.insert(nums[i])
            m.remove(nums[i - k])
            print(m, m.median())
            res.append(m.median())

        return res


class Solution_473:
    def makesquare(self, nums: List[int]) -> bool:
        if not nums:
            return False
        s = sum(nums)
        if s % 4 != 0:
            return False
        t = s // 4
        nums.sort(reverse=True)  # 反向排序，可以减少枚举的次数

        def foo(arr, i):
            if i >= len(nums):
                if all(x == t for x in arr):
                    return True
                return False

            for k in range(4):
                if arr[k] + nums[i] <= t:
                    arr[k] += nums[i]
                    if foo(arr, i + 1):
                        return True
                    arr[k] -= nums[i]
            return False

        return foo([0, 0, 0, 0], 0)


class Solution_1106:
    def parseBoolExpr(self, expression: str) -> bool:
        stack = []
        mp = {'t': True, 'f': False}
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
        return mp[stack[0]]


import functools


class Solution_1377:
    def frogPosition(self, n, edges, T, target):
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # t seconds
        # start from 0

        @functools.lru_cache(None)
        def dfs(node, parent, steps):
            if steps == T:
                return +(node == target)
            tot = ans = 0
            for nei in graph[node]:
                if nei == parent:
                    continue
                ans += dfs(nei, node, steps + 1)
                tot += 1  # 这里计算分支的数目
            if tot == 0:
                return +(node == target)
            return ans / float(tot)
            # return Fraction(ans, tot)

        ans = dfs(1, None, 0)
        return ans  # float(ans)


class Solution_1210:

    def minimumMoves(self, grid: List[List[int]]) -> int:
        n = len(grid)
        H, V = 0, 1

        def canRight(x, y, hv):
            if hv == H:
                return y + 2 < n and grid[x][y + 2] == 0
            else:
                return y + 1 < n and grid[x][y + 1] == grid[x + 1][y + 1] == 0

        def canDown(x, y, hv):
            if hv == H:
                return x + 1 < n and grid[x + 1][y] == grid[x + 1][y + 1] == 0
            else:
                return x + 2 < n and grid[x + 2][y] == 0

        def canRotateCW(x, y, hv):
            return hv == H and x + 1 < n and grid[x + 1][y] == grid[x + 1][y + 1] == 0

        def canRotateCCW(x, y, hv):
            return hv == V and y + 1 < n and grid[x][y + 1] == grid[x + 1][y + 1] == 0

        start = (0, 0, H)
        end = (n - 1, n - 2, H)
        cur_level = {start}
        moves = 0
        visited = set()
        while cur_level:
            next_level = set()  # 用 set 可以降重，用队列可能同一个位置被入堆了多次
            for cur in cur_level:
                visited.add(cur)
                x, y, hv = cur
                if canRight(x, y, hv) and (x, y + 1, hv) not in visited:
                    next_level.add((x, y + 1, hv))
                if canDown(x, y, hv) and (x + 1, y, hv) not in visited:
                    next_level.add((x + 1, y, hv))
                if canRotateCW(x, y, hv) and (x, y, V) not in visited:
                    next_level.add((x, y, V))
                if canRotateCCW(x, y, hv) and (x, y, H) not in visited:
                    next_level.add((x, y, H))
            if end in next_level:
                return moves + 1
            cur_level = next_level
            moves += 1
        return -1
