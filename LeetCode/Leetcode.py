import math
from typing import List
import itertools
import collections

class ListNode:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None


class MyList:
    def __init__(self):
        self.head = ListNode(None, None)
        self.head.next = self.head
        self.head.prev = self.head
        self.size = 0

    def get(self, pt):
        return pt.key, pt.val

    def set(self, pt, key=None, val=None):
        if key:
            pt.key = key
        if val:
            pt.val = val

    def is_tail(self, pt):
        return pt == self.head.prev and pt != self.head

    def pop_head(self):
        if self.head.next != self.head:
            return self.remove(self.head.next)
        return None

    def remove(self, pt):
        q = pt.prev
        t = pt.next
        q.next = t
        t.prev = q

        pt.next = None
        pt.prev = None
        self.size -= 1
        return pt

    def insert(self, pt):
        pt.next = self.head
        pt.prev = self.head.prev
        self.head.prev = pt
        pt.prev.next = pt
        self.size += 1
        return pt

    def print(self):
        p = self.head.next
        while p != self.head:
            print("[%s:%s]->" % (p.key, p.val), end='')
            p = p.next
        print('Null')


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = MyList()
        self.key_to_ptr = {}

    def get(self, key: int) -> int:
        p = self.key_to_ptr.get(key, None)
        if not p:
            return -1
        if not self.cache.is_tail(p):
            p = self.cache.remove(p)
            self.cache.insert(p)
        return p.val

    def put(self, key: int, value: int) -> None:
        p = self.key_to_ptr.get(key, None)
        if p:
            if not self.cache.is_tail(p):
                p = self.cache.remove(p)
                self.cache.insert(p)
            self.cache.set(p, key=p.key, val=value)
        else:
            if self.cache.size >= self.capacity:
                pt = self.cache.pop_head()
                self.key_to_ptr.pop(pt.key)
                if pt:
                    del pt
            p = ListNode(key, v=value)
            self.cache.insert(p)
            self.key_to_ptr[key] = p


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

class GraphNode:
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors

    def __repr__(self):
        return '(%s, %s)' % (self.val, self.neighbors)


class Solution:
    def clone(self, node):
        print(node.val, self.m)
        new_node = GraphNode(node.val, [])
        self.m[node] = new_node
        for n in node.neighbors:
            if n not in self.m:
                new_n = self.clone(n)
                self.m[n] = new_n
            new_node.neighbors.append(self.m[n])
        return node

    def cloneGraph_(self, node):
        self.m = {}
        return self.clone(node)

    def cloneGraph(self, node: 'GraphNode') -> 'GraphNode':
        old2new = {}

        def bfs(node):
            que = [node]
            visit = {}
            while que:
                t = que[0]
                que.pop(0)
                if t not in old2new:
                    new_t = GraphNode(t.val, [])
                    old2new[t] = new_t
                visit[t] = True
                for x in t.neighbors:
                    if x not in old2new:
                        new_x = GraphNode(x.val, [])
                        old2new[x] = new_x
                    if old2new[x] not in old2new[t].neighbors:
                        old2new[t].neighbors.append(old2new[x])
                    if x not in visit:
                        que.append(x)

        bfs(node)
        return old2new[node]

    @staticmethod
    def run():
        s = Solution()
        one = GraphNode(1, [])
        two = GraphNode(2, [])
        thr = GraphNode(3, [])
        fur = GraphNode(4, [])
        one.neighbors = [two, fur]
        two.neighbors = [one, thr]
        thr.neighbors = [two, fur]
        fur.neighbors = [one, thr]
        print('#' * 100)
        s.cloneGraph(fur)
        print('#' * 100)

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __repr__(self):
        return f'{{ {self.val}, left:{{{self.left}}}, right{{{self.right}}}}}'


class Tree:
    def __init__(self, root):
        self.root = root

    def pre_recursive(self):
        res = []

        def pre(root):
            nonlocal res
            if root:
                res += [root.val]
                pre(root.left)
                pre(root.right)

        pre(self.root)
        return res

    def pre_non_recursive(self):
        stack = []
        res = []
        p = self.root
        while stack or p:
            while p:
                res += [p.val]
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                p = p.right
        return res

    def in_recursive(self):
        res = []

        def in_order(root):
            nonlocal res
            if root:
                in_order(root.left)
                res += [root.val]
                in_order(root.right)

        in_order(self.root)
        return res

    def in_non_recursive(self):
        stack = []
        res = []
        p = self.root
        while stack or p:
            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                res += [p.val]
                p = p.right
        return res

    def post_recursive(self):
        res = []

        def post(root):
            nonlocal res
            if root:
                post(root.left)
                post(root.right)
                res += [root.val]

        post(self.root)
        return res

    def post_non_recursive(self):
        res = []
        stack = []
        p = self.root
        last = None
        while stack or p:

            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack[-1]
                if p.right is None or last == p.right:
                    res += [p.val]
                    last = p
                    stack.pop()
                    p = None
                else:
                    p = p.right
        return res

    @staticmethod
    def run():
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        root.right.left = TreeNode(5)
        root.right.right = TreeNode(7)
        tree = Tree(root)
        print('Pre order:')
        print(tree.pre_recursive())
        print(tree.pre_non_recursive())
        print('In order:')
        print(tree.in_recursive())
        print(tree.in_non_recursive())
        print('Post order:')
        print(tree.post_recursive())
        print(tree.post_non_recursive())

class Solution_23:
    # TODO 败者树
    def mergeKLists(self, lists):
        n = len(lists)
        ls = [n] * (n+1)

        def adjust(s, buf):
            """
            :param s:  存胜者
            :param buf:
            :return:
            """
            t = (s+n) // 2
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
        ps = []
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
                b = stack[-1]
                a = stack[-2]
                stack.pop()
                stack.pop()
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
                b = stack[-1]
                a = stack[-2]
                stack.pop()
                stack.pop()
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

class Solution:
    def reverseBetween(self, head, m, n):

        if m == n:
            return head
        new_head = ListNode(None)
        new_head.next = head
        prev = q = p = new_head
        cnt = 0
        while p and cnt <= n:

            if cnt < m:
                prev = q = p
                p = p.next
            elif cnt <= n:
                t = p.next
                p.next = q
                q = p
                p = t
            cnt += 1

        prev.next.next = p
        prev.next = q

        return new_head.next

class TrieNode:
    def __init__(self):
        self.child = [None] * 26
        self.end_of_word = False


class Trie_208:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        p = self.root
        for c in word:
            ix = ord(c.lower()) - ord('a')
            if p.child[ix] is None:
                p.child[ix] = TrieNode()
            p = p.child[ix]
        p.end_of_word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        p = self.root
        for c in word:
            ix = ord(c.lower()) - ord('a')
            if p.child[ix] is None:
                return False
            p = p.child[ix]
        return p.end_of_word

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        p = self.root
        for c in prefix:
            ix = ord(c.lower()) - ord('a')
            if p.child[ix] is None:
                return False
            p = p.child[ix]
        return True


class Solution_54:
    def spiralOrder(self, matrix):
        res = []
        m = len(matrix)
        if m == 0:
            return []
        n = len(matrix[0])
        if n == 0:
            return []

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


class Solution_850:
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
        for n in range(1, len(rectangles)+1):
            for group in itertools.combinations(rectangles, n):
                res += (-1) ** (n+1) * area(reduce(interscet, group))
                pass

        return res % int(1e9 + 7)

    def rectangleArea(self, rectangles: "List[List[int]]") -> int:
        pass



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

    # TODO this method also solved 587
    @staticmethod
    def convex_hull(points):

        def ccw(p1, p2, p3):
            # a = p3 - p1
            # b = p2 - p1
            # a X b = |a|*|b|sin<a,b> > 0 则b在a的逆时针方向
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


class MyHashSet_705:
    # TODO 开方定址法，平方探测
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.primes = 10007
        self.hash_table = [None] * self.primes

    def add(self, key: int) -> None:
        idx = key % self.primes
        i = 1
        sign = 1
        while self.hash_table[idx] != None:
            if self.hash_table[idx] == key:
                return
            idx = (key + i * i * sign) % self.primes
            if sign == -1:
                i += 1
            sign *= -1

        self.hash_table[idx] = key

    def remove(self, key: int) -> None:
        idx = key % self.primes
        i = 1
        sign = 1
        while self.hash_table[idx] != None and self.hash_table[idx] != key:
            idx = (key + i * i * sign) % self.primes
            if sign == -1:
                i += 1
            sign *= -1
        self.hash_table[idx] = None

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        idx = key % self.primes
        i = 1
        sign = 1
        while self.hash_table[idx] != None:
            if self.hash_table[idx] == key:
                return True
            idx = (key + i * i * sign) % self.primes
            if sign == -1:
                i += 1
            sign *= -1
        return False


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


class Solution_235:
    def _0_lowestCommonAncestor(self, root, p, q):
        """
        using bool flags to track whether we have encountered the nodes(p and q).
        if we have travel through both nodes, say, two of the flags was set True,
        then we find our answer.

        Time complexity: O(n)
        Space complexity: O(n)
        """
        ans = None

        def post(root):
            nonlocal ans
            if not root: return False
            left = post(root.left)
            right = post(root.right)
            mid = root == p or root == q
            if mid + left + right >= 2:
                ans = root
            return mid or left or right

        post(root)
        return ans

    def _1_lowestCommonAncestor(self, root, p, q):
        """using parent pointer"""
        parent = {root: None}
        stack = [root]
        while p not in parent or q not in parent:
            t = stack.pop()
            if t.left:
                parent[t.left] = t
                stack.append(t.left)
            if t.right:
                parent[t.right] = t
                stack.append(t.right)
        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]

        while q not in ancestors:
            q = parent[q]
        return q

    def _2_lowestCommonAncestor(self, root, p, q):
        stack = [(root, 0)]  # 0: unsearched, 1: left searched, 2: both searched
        one_found = False
        LCA_index = -1
        while stack:
            t, t_state = stack[-1]
            if t_state == 0:
                if t == p or t == q:
                    if one_found:
                        return stack[LCA_index][0]
                    else:
                        one_found = True
                        LCA_index = len(stack) - 1
                stack.pop()
                stack.append((t, t_state + 1))
                if t.left:
                    stack.append((t.left, 0))
            elif t_state == 1:
                stack.pop()
                stack.append((t, t_state + 1))
                if t.right:
                    stack.append((t.right, 0))
            else:
                if one_found and LCA_index == len(stack) - 1:
                    LCA_index -= 1
                stack.pop()
        return None

    def _3_lowestCommonAncestor(self, root, p, q):
        def foo(root):
            if p.val > root.val and q.val > root.val:  # 这里决定了一定有右子树
                return foo(root.right)
            elif p.val < root.val and q.val < root.val:  # 这里决定了一定有左子树
                return foo(root.left)
            else:
                return root

        return foo(root)

    def _lowestCommonAncestor(self, root, p, q):
        t = root
        while t:
            if p.val > t.val and q.val > t.val:
                t = t.right
            elif p.val < t.val and q.val < t.val:
                t = t.left
            else:
                return t

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > q.val:
            p, q = q, p

        def post(root):
            if not root: return None
            rleft = post(root.left)
            rright = post(root.right)
            if rleft or rright:
                if p.val <= root.val <= q.val:
                    return root
                else:
                    return rleft or rright
            else:
                if p.val <= root.val <= q.val:
                    return root
                else:
                    return None

        return post(root)


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


class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = ListNode(None)
        self.tail = self.head
        self.size = 0

    def _print_(self):
        p = self.head.next
        while p:
            print(p.val, end=' ')
            p = p.next
        print('')

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        print('get', index, self.size)
        self._print_()

        if index >= self.size:
            return -1
        i = -1
        p = self.head
        while p:
            p = p.next
            i += 1
            if i == index:
                return p.val
        return -1

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list.
        After the insertion, the new node will be the first node of the linked list.
        """
        p = ListNode(val)
        p.next = self.head.next
        self.head.next = p
        if self.tail == self.head:
            self.tail = p
        self.size += 1

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        print('tail=', self.tail.val)
        p = ListNode(val)
        p.next = self.tail.next
        self.tail.next = p
        self.tail = p
        self.size += 1
        self._print_()

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list.
        If index equals to the length of linked list, the node will be appended
        to the end of linked list. If index is greater than the length, the
        node will not be inserted.
        """
        if index == self.size:
            self.addAtTail(val)
        elif index < self.size:
            i = -1
            q = p = self.head
            while p:
                q = p
                p = p.next
                i += 1
                if i == index:
                    break
            if q:
                node = ListNode(val)
                node.next = q.next
                q.next = node
                if self.tail == self.head:
                    self.tail = node
                self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < self.size:
            i = -1
            q = p = self.head
            while p:
                q = p
                p = p.next
                i += 1
                if i == index:
                    break
            if q and p:
                if p == self.tail:
                    self.tail = q
                q.next = p.next
                p.next = None
                del p
                self.size -= 1


class Solution_937:
    def reorderLogFiles(self, logs):
        # TODO 一个特别fancy 的想法，使用tuple来排序，然后sorted和sorted 都是稳定的
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
                f = 0  # TODO f的初始值需要 再确定，不过貌似初始值不影响
                j = m - 1
                for i in range(m - 2, -1, -1):
                    if i > j and suff[i + m - 1 - f] < i - j:
                        suff[i] = suff[i + m - 1 - f]
                    else:
                        # j = min(j, i)
                        if i < j:
                            j = i
                        f = i
                        while j >= 0 and pattern[j] == pattern[j + m - f]:
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
              前缀和后缀abab,长度为PMT[j-1] = 4, 因此这一部分不需要再匹配，直接移动到PMT[j-1]与i匹配即可（i不需要移动）
         实际编程中，为了方便，常将PMT数组右移一位，0位置设为-1（编程方便），记为next数组，当出现失配时，移动到next[j]进行
         匹配。

         求next数组（或者PMT数组）是找模式串前缀与后缀匹配的最长的长度，因此也是一个字符串匹配过程，即以模式字符串为主
         字符串，以模式字符串的前缀为目标字符串，一旦字符串匹配成功，那么当前的next值就是匹配成功的字符串的长度。匹配时，
         作为目标的模式串index[1, len-1], 作为模式的模式串index[0, len-2], 即保证前缀与后缀匹配
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
    # link to 338, 不太一样
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
        st = SegmentTree(heights)

        def maxArea(l, r):
            if l > r:
                return -float('inf')
            elif l == r:
                return st.ls[l]
            m = st.RMQ(l, r)
            return max(maxArea(l, m - 1), maxArea(m + 1, r), (r - l + 1) * st.ls[m])

        return maxArea(0, len(st.ls) - 1)

    def _largestRectangleArea(self, heights: 'List[int]') -> int:
        stack = []
        m = 0
        for i in range(len(heights)):
            print(stack)
            while stack and heights[stack[-1]] >= heights[i]:
                area = heights[stack.pop()] * (i if len(stack) == 0 else i - 1 - stack[-1])
                m = max(m, area)
            stack.append(i)

        while stack:
            area = heights[stack.pop()] * (len(heights) if len(stack) == 0 else len(heights) - 1 - stack[-1])
            m = max(m, area)

        return m


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


class Solution_140:
    def _brute_force_TLE_wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        res = []
        memo = {}
        for w in wordDict:
            memo[w] = memo.get(w, 0) + 1

        def _dfs(memo, lo, hi, tmp):
            nonlocal res
            if hi == len(s):
                if s[lo:hi] in memo:
                    tmp += [s[lo:hi]]
                    res.append(tmp)
            elif hi < len(s):
                if s[lo:hi + 1] in memo:
                    _dfs(memo, hi + 1, hi + 1, tmp + [s[lo:hi + 1]])
                _dfs(memo, lo, hi + 1, tmp)

        _dfs(memo, 0, 0, [])
        return [' '.join(x) for x in res]

    def _dfs_wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        words = set(wordDict)

        def search(s, part_memo):
            res = []
            for i in range(len(s)):
                left = s[:i + 1]
                if left in words:
                    right = s[i + 1:]
                    if len(right) == 0:
                        res.append([left])
                        return res
                    if right not in part_memo:
                        part_memo[right] = search(right, part_memo)  # TODO trik是把已经搜索过的结果记下来（带memo的搜素）
                    for ls in part_memo[right]:
                        res.append([left] + ls)
            return res

        res = search(s, {})
        return [' '.join(ls) for ls in res]

    def _trie_memo_wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        trie = {}
        for w in wordDict:
            p = trie
            for c in w:
                if c not in p:
                    p[c] = {}
                p = p[c]
            p['#'] = w

        def dfs_find(s, memo):
            res = []
            p = trie
            for i, c in enumerate(s):
                if c not in p:
                    break
                p = p[c]
                if '#' in p:
                    right = s[i + 1:]
                    if len(right) == 0:
                        res.append([p['#']])
                        return res
                    if right not in memo:
                        memo[right] = dfs_find(right, memo)  # 递归查找
                    for ls in memo[right]:
                        res.append([p['#']] + ls)
            return res

        res = dfs_find(s, {})
        return [' '.join(ls) for ls in res]

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        """DP and build path"""
        mlen = len(max(wordDict, key=len))
        memo = set(wordDict)
        res = []
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in memo:
                    dp[i] = 1

        def build_path(left, idx):
            nonlocal res, dp, memo
            if idx == 0:
                res.append(left[:])
                return
            for j in range(max(0, idx - mlen), idx):
                w = s[j:idx]
                if dp[j] and w in memo:
                    left.insert(0, w)
                    build_path(left, j)
                    left.pop(0)

        build_path([], len(s))
        return [' '.join(ls) for ls in res]


class Solution_472:
    def _findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        def check(w, mm):
            for i in range(len(w)):
                left, right = w[:i + 1], w[i + 1:]
                if left in mm:
                    if right in mm or check(right, mm):
                        return True
            return False

        memo = set([w for w in words if len(w)])
        res = []
        for w in words:
            if len(w) > 0:
                if check(w, memo):
                    res.append(w)
        return res

    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:

        def create_trie(words):
            root = {}
            for w in words:
                if len(w) == 0: continue
                p = root
                for c in w:
                    if c not in p:
                        p[c] = {}
                    p = p[c]
                p['#'] = w
            return root

        def check(root, w):
            if len(w) == 0:
                return False
            p = root
            for c in w:
                if c not in p:
                    return False
                p = p[c]
            return '#' in p

        def search(root, word):
            for i in range(len(word)):
                left, right = word[:i + 1], word[i + 1:]
                if check(root, left):
                    if check(root, right) or search(root, right):
                        return True
            return False

        root = create_trie(words)
        res = []
        for w in words:
            if len(w) > 0 and search(root, w):
                res.append(w)
        return res

    def _DP_findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:

        def check(word, memo):
            # TODO this also solve 139 word break， and aslo using for 140 (DP + rebuild)
            dp = [0] * (len(word) + 1)
            dp[0] = 1
            for i in range(1, len(word) + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in memo:
                        dp[i] = 1
                        break
            return dp[len(word)]

        memo = set(words)
        res = []
        for w in words:
            if len(w) > 0:
                memo.remove(w)
                if check(w, memo):
                    res.append(w)
                memo.add(w)
        return res


class Solution_10:
    def _isMatch(self, s: str, p: str) -> bool:
        def match(t, p):
            if len(p) == 0:
                return not len(t)
            r = (len(t) and (p[0] == t[0] or p[0] == '.'))
            if len(p) >= 2 and p[1] == '*':
                return match(t, p[2:]) or (r and match(t[1:], p))
            else:
                return r and match(t[1:], p[1:])

        return match(s, p)

    def _DP_isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = True
        # 处理 a*, .*, a*b* 这种特殊情况
        for i in range(n):
            if p[i] == '*':
                dp[0][i + 1] = dp[0][i - 1]
        for i, cs in enumerate(s):
            for j, cp in enumerate(p):
                if cp in {cs, '.'}:
                    dp[i + 1][j + 1] = dp[i][j]
                elif cp == '*':
                    print(dp[i + 1][j - 1])
                    dp[i + 1][j + 1] = dp[i + 1][j + 1] or dp[i + 1][j - 1]
                    if p[j - 1] in {cs, '.'}:
                        dp[i + 1][j + 1] = dp[i + 1][j + 1] or dp[i][j + 1]
                else:
                    dp[i + 1][j + 1] = False
        for x in dp:
            print(list(map(int, x)))
        return dp[m][n]

    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [False for _ in range(n + 1)]
        dp[0] = True
        for i in range(n):
            if p[i] == '*':
                dp[i + 1] = dp[i - 1]
        # print(list(map(int, dp)))
        for i, cs in enumerate(s):
            new_dp = [False for _ in range(n + 1)]
            for j, cp in enumerate(p):
                if cp in {cs, '.'}:
                    new_dp[j + 1] = dp[j]
                elif cp == '*':
                    new_dp[j + 1] = new_dp[j - 1]
                    if p[j - 1] in {cs, '.'}:
                        new_dp[j + 1] = new_dp[j + 1] or dp[j + 1]
                else:
                    new_dp[j + 1] = False
            # print(list(map(int, new_dp)))
            dp[:] = new_dp
        # for x in dp:
        #     print(list(map(int, x)))
        return dp[n]

    def __isMatch(self, text, pattern):
        # 反向
        dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]

        dp[-1][-1] = True
        for i in range(len(text), -1, -1):
            for j in range(len(pattern) - 1, -1, -1):
                first_match = i < len(text) and pattern[j] in {text[i], '.'}
                if j + 1 < len(pattern) and pattern[j + 1] == '*':
                    dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
                else:
                    dp[i][j] = first_match and dp[i + 1][j + 1]

        return dp[0][0]


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


# a = [341,22,175,424,574,687,952,439,51,562,962,890,250,47,945,914,835,937,419,343,125,809,807,959,403,861,296,39,802,562,811,991,209,375,78,685,592,409,369,478,417,162,938,298,618,745,888,463,213,351,406,840,779,299,90,846,58,235,725,676,239,256,996,362,819,622,449,880,951,314,425,127,299,326,576,743,740,604,151,391,925,605,770,253,670,507,306,294,519,184,848,586,593,909,163,129,685,481,258,764]
# b = [462,101,820,999,900,692,991,512,655,578,996,979,425,893,975,960,930,991,987,524,208,901,841,961,878,882,412,795,937,807,957,994,963,716,608,774,681,637,635,660,750,632,948,771,943,801,985,476,532,535,929,943,837,565,375,854,174,698,820,710,566,464,997,551,884,844,830,916,970,965,585,631,785,632,892,954,803,764,283,477,970,616,794,911,771,797,776,686,895,721,917,920,975,984,996,471,770,656,977,922]
# c = [85,95,14,72,17,3,86,65,50,50,42,75,40,87,35,78,47,74,92,10,100,29,55,57,51,34,10,96,14,71,63,99,8,37,16,71,10,71,83,88,68,79,27,87,3,58,56,43,89,31,16,9,49,84,62,30,35,7,27,34,24,33,100,25,90,79,58,21,31,30,61,46,36,45,85,62,91,54,28,63,50,69,48,36,77,39,19,97,20,39,48,72,37,67,72,46,54,37,53,30]
la = [1, 2, 3, 3]
lb = [3, 4, 5, 6]
lc = [50, 10, 40, 70]

print(Solution_5233().jobScheduling(la, lb, lc))
print(120)


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
            # print(t, ia, ib, ic)

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
        def gcd(a, b):
            while b > 0:
                a, b = b, a % b
            return a

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
            return cuts < m  # (cuts + 1) < m 最后cur中还剩一个子集。

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
        # 有 x_{k+1} - x_k = f(x_k) / f'(x_k)
        #    x_{k+1} = (x_k ^2 + n)/ 2x_k

        r = x
        while r * r > x:
            r = (r + x // r) // 2
        return r

class Solution_887:
    def superEggDrop(self, K: int, N: int) -> int:
        # DP(memo+search) + binary search
        # 状态(K, N)
        # 如果我们从X层丢鸡蛋，碎了状态变成(K-1, X-1), 没碎状态变成(K, N-X)
        # 定义dp(k, n) = 在状态(k, n)下解这个问题需要的最多的步数，则
        #   dp(k, n) = min_{1<=X<=N} (max(dp(k-1, x-1), dp(k, n-x)))
        # 注意到 t1 = dp(k-1, x-1), 随x单调增
        #        t2 = dp(k, n-x), 随x单调减
        # 于是max(t1,t2)是二者的上半部分, 因此查找x可以二分进行。
        # 边界条件是两者不一定相交于整数X, 因此这时要检查两个。
        # Time: O(KN log N)
        # space: O(KN)
        # https://leetcode.com/articles/super-egg-drop/ Solution1
        dp = {}

        def foo(k, n):
            if (k, n) in dp:
                return dp[(k, n)]

            if n == 0:
                ans = 0
            elif k == 1:
                ans = n
            else:
                lo, hi = 1, n
                while lo + 1 < hi:
                    x = (lo + hi) >> 1
                    t1 = foo(k - 1, x - 1)
                    t2 = foo(k, n - x)

                    if t1 < t2:
                        lo = x
                    elif t1 > t2:
                        hi = x
                    else:
                        lo = hi = x
                ans = 1 + min(max(foo(k - 1, x - 1), foo(k, n - x)) for x in (lo, hi))
            dp[(k, n)] = ans
            return dp[(k, n)]

        return foo(K, N)

    def superEggDrop(self, K: int, N: int) -> int:
        # 使用上述定义  dp(k, n) = min_{1<=X<=N} (max(dp(k-1, x-1), dp(k, n-x)))
        #   t1 = dp(k-1, x-1)，注意到t1随x单调增，但是与 n 无关
        #   t2 = t2 = dp(k, n-x), 随x单调减, 但随 n 单调增
        # 于是可以得到 dp(k, n) 随 n 的增加而增加。https://LeetCode.com/articles/super-egg-drop/ 这里有图
        # Time: O(kN)
        # space: O(N)
        dp = list(range(N+1))
        for k in range(2, K+1):
            print(dp)
            dp_tmp = [0]  # dp_tmp = dp(k, ·)
            x = 1
            for n in range(1, N+1):
                # Increase our optimal x while we can make our answer better.
                while x < n and max(dp[x-1], dp_tmp[n-x]) > max(dp[x], dp_tmp[n-x-1]):
                    x += 1
                    print('#')
                dp_tmp.append(1 + max(dp[x-1], dp_tmp[n-x]))
            dp = dp_tmp
        return dp[-1]

    def superEggDrop(self, K, N):
        # 反向思考这个问题，假如给出步数T，K个鸡蛋,，f(T,K)表示我们能够解原问题的楼层数的最大值。
        # （能够解原问题指：找到0<=F<=f(T,K)即确定找到楼层F).
        # 则问题转换为寻找满足 f(T,K) >= N 的T的最小值。即 min T s.t. f(T, K) >= N
        # 在最优策略下，我们在解X' 层丢一个鸡蛋，如果破了，我们可以解f(T-1, K-1), 如果没有碎，可以解f(T-1, K)
        # 于是   f(T, K) = 1 + f(T-1, K-1) + f(T-1, K) 且显然f(t, 1) = t (t>=1), f(1, k) = 1(k>=1)
        # 接下来用两种方式解的f(T, K)的通项
        #    i). 记g(t, k) = f(T, K) - F(T, K-1)
        #          f(T, K)    = 1 + f(T-1, K-1) + f(T-1, K)
        #          f(T, K-1)  = 1 + f(T-1, K-2) + f(T-1, K-1)
        #          g(t, k) = f(T, K) - f(T, K-1) = f(T-1, k) - f(T-1, K-2) = g(t-1, k) + g(t-1, k-1)
        #          上式子 g(t, k) = g(t-1, k) + g(t-1, k-1) 是一个二项分布的递归式，其解为g(t, k) = C(t, k+1)
        #        则：f(t, k) = \sum_{1<=x<=K} g(t, x) = \sum C(t, x)
        #
        #   ii). 另一个角度来看，我们有t次尝试和k个鸡蛋，因此这是一个长度为t,失败（鸡蛋碎）次数最多为k的尝试序列。
        #        没有失败的是C(n, 0), 一次失败是C(n, 1)..., 综合起来就是\sum C(t, x)
        #
        #  by using  C(n, k + 1) = C(n, k) * (n-k)/(k+1) 可以简化计算

        def combination(x):
            # C(n, k + 1) = C(n, k) * (n-k)/(k+1)
            ans = 0
            r = 1
            for i in range(1, K+1):
                r *= x - i + 1   # r = r * (x - (i-1)) // ((i-1) + 1)
                r //= i
                ans += r
                if ans >= N:
                    break
            return ans

        lo, hi = 1, N
        while lo < hi:
            mi = (lo + hi) >> 1
            if combination(mi) < N:
                lo = mi + 1
            else:
                hi = mi
        return lo


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
              10: 'X', 40: 'XL', 50: 'L', 90: 'XC',
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


class Solution_57:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        def intersect(in1, in2):
            a, b = in1
            c, d = in2
            return a <= d and c <= b

        i = 0
        inserted = False
        inter = False
        while i < len(intervals):
            if intervals[i][0] > newInterval[0] and not intersect(intervals[i], newInterval):
                intervals.insert(i, newInterval)
                inserted = True
                break
            if intersect(intervals[i], newInterval):
                (a, b), (c, d) = intervals[i], newInterval
                intervals[i] = [min(a, c), max(b, d)]
                inter = True
                break
            i += 1
        if inter:
            while i < len(intervals) - 1:
                if intersect(intervals[i], intervals[i + 1]):
                    (a, b), (c, d) = intervals[i], intervals[i + 1]
                    intervals[i] = [min(a, c), max(b, d)]
                    intervals.pop(i + 1)
                else:
                    i += 1
        elif not inserted:
            intervals.append(newInterval)

        return intervals

    def _insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        def intersect(in1, in2):
            a, b = in1
            c, d = in2
            return a <= d and c <= b

        i = 0
        inter = False
        while i < len(intervals):
            if intersect(intervals[i], newInterval):
                (a, b), (c, d) = intervals[i], newInterval
                intervals[i] = [min(a, c), max(b, d)]
                inter = True
                break
            i += 1
        if inter:
            while i < len(intervals) - 1:
                if intersect(intervals[i], intervals[i + 1]):
                    (a, b), (c, d) = intervals[i], intervals[i + 1]
                    intervals[i] = [min(a, c), max(b, d)]
                    intervals.pop(i + 1)
                else:
                    i += 1
        else:
            inserted = False
            for i in range(len(intervals)):
                if intervals[i][0] > newInterval[0]:
                    intervals.insert(i, newInterval)
                    inserted = True
                    break
            if not inserted:
                intervals.append(newInterval)

        return intervals


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
for x in [4, 3,2,1]:
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
        r = 0
        for x in nums:
            r ^= x
        return r


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
                x &= (x-1)
                cnt += 1
            return cnt
        for i in range(num+1):
            res.append(count(i))
        return res

    def countBits(self, num: int) -> List[int]:
        def gen(n):
            if n == 2:
                return [0, 1]
            t = gen(n//2)
            return t + [x+1 for x in t]

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
            t =nextPow2(num)
        ls = gen(t)
        return ls[:num+1]

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
        # 先存一遍右边的最高点
        right = [0] * (len(height) + 1)
        for i in range(len(height) - 1, -1, -1):
            right[i] = max(right[i + 1], height[i])
        right = right[1:]

        lm = 0
        res = 0
        for i in range(len(height)):
            t = min(lm, right[i]) - height[i]
            if t > 0:
                res += t
            lm = max(lm, height[i])
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
            board[x][y] = 'X'

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


class Solution_121:
    def maxProfit(self, prices: List[int]) -> int:
        # 因为可以任意多个transaction, 所以有利润就加上去
        profit = 0
        for i in range(len(prices)-1):
            t = prices[i+1] - prices[i]
            if t > 0:
                profit += t
        return profit


class Solution_109:
    def _sortedListToBST(self, head: ListNode) -> TreeNode:
        ls = []
        p = head
        # 把节点存在数组里，也可以用快慢指针来找中点
        while p:
            ls.append(p.val)
            p = p.next

        def build_tree(inorder):
            if len(inorder) == 0:
                return None
            mid = len(inorder) // 2
            node = TreeNode(inorder[mid])
            node.left = build_tree(inorder[:mid])
            node.right = build_tree(inorder[mid + 1:])
            return node

        return build_tree(ls)

    def sortedListToBST(self, head: ListNode) -> TreeNode:
        # solution 3给出的方法太牛皮了
        n = 0
        p = head
        while p:
            n += 1
            p = p.next

        def convert(lo, hi):
            nonlocal head
            if lo > hi:
                return None

            mid = (lo + hi) >> 1
            left = convert(lo, mid - 1)  # 先向左递归

            node = TreeNode(head.val)
            node.left = left

            head = head.next

            node.right = convert(mid + 1, hi)
            return node

        return convert(0, n - 1)

    @staticmethod
    def run():
        head = ListNode(-10)
        head.next = ListNode(-5)
        head.next.next = ListNode(0)
        head.next.next.next = ListNode(5)
        head.next.next.next.next = ListNode(10)

        print(Solution_109().sortedListToBST(head))

class Solution_95:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def gen_tree(lo, hi):
            if lo > hi:
                return [None]
            res = []
            for i in range(lo, hi+1):
                left_ls = gen_tree(lo, i-1)
                right_ls = gen_tree(i+1, hi)
                for left in left_ls:
                    for right in right_ls:
                        root = TreeNode(i)
                        root.left = left
                        root.right = right
                        res.append(root)
            return res
        if n < 1:
            return []
        return gen_tree(1, n)


class Codec_449:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        pre = []
        ino = []

        def pre_order(root):
            nonlocal pre
            if root:
                pre.append(root.val)
                pre_order(root.left)
                pre_order(root.right)

        def in_order(root):
            nonlocal ino
            if root:
                in_order(root.left)
                ino.append(root.val)
                in_order(root.right)

        pre_order(root)
        in_order(root)
        # print(pre)
        # print(ino)
        return str(pre) + '##' + str(ino)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        pre, ino = data.split('##')
        if len(pre) <= 2:
            return None
        pre = [int(x) for x in pre[1:-1].split(',')]
        ino = [int(x) for x in ino[1:-1].split(',')]

        # print(pre)
        # print(ino)
        def build(pre, ino):
            if len(pre) == 0:
                return None
            root = TreeNode(pre[0])
            idx = ino.index(pre[0])
            n = idx
            root.left = build(pre[1:n + 1], ino[:idx])
            root.right = build(pre[n + 1:], ino[idx + 1:])
            return root

        return build(pre, ino)


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        def level(root):
            que = collections.deque([root])
            res = []
            while que:
                t = que.popleft()
                if t:
                    res.append(t.val)
                    que.append(t.left)
                    que.append(t.right)
                else:
                    res.append(None)

            while res and res[-1] == None:
                res.pop()
            return res

        lv = level(root)

        # print(lv)
        return str(lv)[1:-1]

    def deserialize(self, data):
        if len(data) == 0:
            return None
        data = data.replace(' ', '')
        nodes = [None if x == 'None' else TreeNode(int(x))
                 for x in data.split(',')]
        kids = nodes[::-1]
        root = kids.pop()
        for node in nodes:
            if node:
                if kids: node.left = kids.pop()
                if kids: node.right = kids.pop()
        return root

    def _deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if len(data) == 0:
            return None
        lv = [int(x) if x != ' None' else None for x in data.split(',')]
        # print(lv)

        que = collections.deque()
        root = TreeNode(lv[0])
        que.append(root)
        idx = 1
        while que:
            t = que.popleft()
            if idx < len(lv):
                if lv[idx] == None:
                    t.left = None
                else:
                    t.left = TreeNode(lv[idx])
                    que.append(t.left)
            if idx + 1 < len(lv):
                if lv[idx + 1] == None:
                    t.right = None
                else:
                    t.right = TreeNode(lv[idx + 1])
                    que.append(t.right)
            idx += 2

        return root
