# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
        
        if not root1 or not root2:
            return root1 == root2 # both None-> true, otherwis false
        def bfs(root):
            d = {}
            q = [root]
            while len(q)!=0:
                t = q.pop(0)
                if t.left:
                    q.append(t.left)
                    d[t.left.val] = t.val
                if t.right:
                    q.append(t.right)
                    d[t.right.val] = t.val
            return d
        
        return bfs(root1) == bfs(root2)
    
    def _flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
        def pre(root1, root2):
            if not root1 or not root2:
                return root1 == root2 # both None-> true, otherwis false
            if root1.val != root2.val:
                return False
            p = pre(root1.left, root2.left)
            q = pre(root1.left, root2.right)

            m = pre(root1.right, root2.left)
            n = pre(root1.right, root2.right)
            return (p or q) and (m or n)
        return pre(root1, root2)