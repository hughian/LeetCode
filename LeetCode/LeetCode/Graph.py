
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