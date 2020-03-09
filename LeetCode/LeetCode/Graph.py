from typing import List
import collections
import itertools


class GraphNode:
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors

    def __repr__(self):
        return '(%s, %s)' % (self.val, self.neighbors)


class Solution_133:
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
        s = Solution_133()
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


class Solution_399:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 图问题，有向图，value 是边上的权重
        # eval 就是起始点到目标点路径上权重的积
        # 没有路径连通就是 -1.0
        G = collections.defaultdict(list)
        for (a, b), v in zip(equations, values):
            G[a].append([b, v])
            G[b].append([a, 1. / v])

        res = []
        for a, b in queries:
            # 找从 a 出发的路径 BFS
            if a not in G or b not in G:
                res.append(-1.0)
                continue
            visit = {a}
            que = collections.deque([(a, 1)])
            while que:
                t, val = que.popleft()
                if t == b:
                    break
                for c, v in G[t]:
                    if c not in visit:
                        visit.add(c)
                        que.append([c, val * v])

            res.append(val if t == b else -1.0)
        return res


class Solution_785:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # 二部图
        def bfs(i, visit):
            s1, s2 = set(), set()
            que = collections.deque([i])
            visit[i] = 1
            lv = 0
            while que:
                L = len(que)
                lv += 1
                for _ in range(L):
                    t = que.popleft()
                    if lv & 1:
                        s1.add(t)
                    else:
                        s2.add(t)
                    for v in graph[t]:
                        if visit[v] == 0:
                            visit[v] = 1
                            que.append(v)
            # print(s1, s2)
            for s in [s1, s2]:
                for a, b in itertools.combinations(s, 2):
                    if b in graph[a] or a in graph[b]:
                        return False
            return True

        visit = [0] * len(graph)

        for i in range(len(graph)):
            if visit[i] == 0:
                if not bfs(i, visit):
                    return False
        return True


######################################################################################################################
# Union Find
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        a = self.find(x)
        b = self.find(y)
        self.p[a] = b


class Solution_959:
    def regionsBySlashes(self, grid: List[str]) -> int:
        # 难点在于定义图，可以用东西南北四个三角形（两个对角线分割开）的表示一个正方形
        # 使用并查集
        # +-------+
        # | \ 0 / |
        # |1  X  2|
        # | / 3 \ |
        # +-------+
        N = len(grid)
        uf = UnionFind(4 * N * N)
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                root = 4 * (r * N + c)

                if val in '/ ':  # '/' 或者 ' '
                    uf.union(root + 0, root + 1)
                    uf.union(root + 2, root + 3)
                if val in '\\ ':  # '\\' 或者 ' ' 这里不用 elif
                    uf.union(root + 0, root + 2)
                    uf.union(root + 1, root + 3)

                if r + 1 < N:
                    uf.union(root + 3, (root + 4 * N) + 0)  # 下边合并到下一行上边
                if r - 1 >= 0:
                    uf.union(root + 0, (root - 4 * N) + 3)  # 上边合并到上一行下边

                if c + 1 < N:
                    uf.union(root + 2, (root + 4) + 1)  # 合并到右边一列左边
                if c - 1 >= 0:
                    uf.union(root + 1, (root - 4) + 2)  # 合并到左边一列右边

        return sum(uf.find(x) == x for x in range(4 * N * N))
