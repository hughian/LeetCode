# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        visited = {}
        parents = {}
        def pre(root, parent):
            nonlocal parents
            if root:
                visited[root] = False
                parents[root] = parent
                pre(root.left, root)
                pre(root.right, root)
        pre(root, None)
        
        def bfs(node, k):
            que = [node, None]
            ans = []
            dist = 0
            while len(que) > 1:
                t = que.pop(0)
                if t is None:
                    dist += 1
                    que.append(None)
                    continue
                visited[t] = True
                if dist == k and t is not None:
                    ans += [t.val]
                
                for n in [t.left, t.right, parents[t]]:
                    if n and not visited[n]:
                        que.append(n)
            return ans
        
        return bfs(target, K)

def level(root):
    que = [root]
    ans = []
    while que:
        t = que.pop(0)
        if t: ans.append(t.val)
        else:
            ans.append(None)
            continue
        que.append(t.left)
        que.append(t.right)
    while not ans[-1]:
        ans.pop()
    return ans

root = TreeNode(3)
root.left = TreeNode(5)
root.right = TreeNode(1)
root.left.left = TreeNode(6)
root.left.right = TreeNode(2)
root.left.right.left = TreeNode(7)
root.left.right.right = TreeNode(4)
root.right.left = TreeNode(0)
root.right.right = TreeNode(8)

    
s = Solution()
r = s.distanceK(root, root.left, 2)
print(r)

