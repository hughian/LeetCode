# 并查集
def frc(edges):
        a = [-1 for _ in range(1001)]
        def root(v):
            x = v
            while a[x] != -1:
                x = a[x]
            return x
        
        for e in edges:
            v1, v2 = e
            if root(v2) == root(v1):
                return e
            a[root(v2)] = root(v1)
            print(e, a[1:11])
edges = [[9,10],[5,8],[2,6],[1,5],[3,8],[4,9],[8,10],[4,10],[6,8],[7,9]]
print(frc(edges))
