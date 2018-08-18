# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def preOrder(self,p,q):
        if self.flag==False:return
        if (not p and q) or (p and not q): 
            self.flag=False
            return
        if not p and not q: return
        if p.val != q.val: self.flag=False
        self.preOrder(p.left,q.left)
        self.preOrder(p.right,q.right)
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        self.flag=True
        self.preOrder(p,q)
        return self.flag
    def isSameTree_(self, p, q):
        if p and q:
            return p.val == q.val and self.isSameTree_(p.left,q.left) and self.isSameTree_(p.right, q.right)
        return p is q # p is q return True if p==None and q==None else False
        
def main():
    p = TreeNode(1)
    p.left = TreeNode(1)
    q = TreeNode(1)
    q.right = TreeNode(1)
    s = Solution().isSameTree(p,q)
    print(s)
    
if __name__=='__main__':
    main()