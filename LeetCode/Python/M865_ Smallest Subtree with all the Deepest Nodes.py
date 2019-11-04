# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        parents = {}
        m = 0
        mleaf = set()
        def pre(root, depth, parent):
            nonlocal m, mleaf, parents
            if root:
                parents[root] = parent
                if root.left == root.right == None:
                    if depth > m:
                        m = depth
                        mleaf = {parent}
                    elif depth == m:
                        mleaf.add(parent)
                pre(root.left, depth+1, root)
                pre(root.right, depth+1, root)
        
        pre(root, 0, None)
        print(mleaf)
        while len(mleaf) > 1:
            new = set()
            for leaf in mleaf:
                new.add(parents[leaf])
            mleaf = new
        return mleaf.pop()

    def subtreeWithAllDeepest(self, root):
        def deep(root):
            if not root: return 0, None
            l, r = deep(root.left), deep(root.right)
            if l[0] > r[0]: return l[0] + 1, l[1]
            elif l[0] < r[0]: return r[0] + 1, r[1]
            else: return l[0] + 1, root
        return deep(root)[1]

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
subt = s.subtreeWithAllDeepest(root)
print(subt.val)

print(level(subt))
