# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def preOrder(root):
    if not root:return
    print(root.val,' ')
    preOrder(root.left)
    preOrder(root.right)
class Solution(object):
    def helper(self, p, q):
        if not p and not q:
            return True
        elif not p or not q:#last if rule out case: p==None and q==None
            return False
        else:
            return p.val==q.val and self.helper(p.left, q.right) and self.helper(p.right, q.left)

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return True
        return self.helper(root,root)
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        que = [root] * 2
        while len(que):
            p = que[0]
            q = que[-1]
            que = que[1:-1]
            if not p and not q: continue
            if not p or not q: return False 
            if p.val != q.val: return False
            que = [p.right] + [p.left] + que + [q.right] + [q.left]
        return True

        
def main():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right= TreeNode(2)
    root.left.left = TreeNode(3)
    root.left.right= TreeNode(4)
    root.right.left= TreeNode(4)
    root.right.right=TreeNode(3)
    
    s = Solution().isSymmetric(root)
    print(s)
    
if __name__=='__main__':
    main()