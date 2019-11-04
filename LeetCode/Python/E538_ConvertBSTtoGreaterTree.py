# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def inOrder(root):
    if not root: return
    inOrder(root.left)
    print(root.val,end=' ')
    inOrder(root.right)

class Solution:
    def in_(self, root):
        if not root: return
        self.in_(root.left)
        self.vals += [root.val]
        self.in_(root.right)
    def inOrder(self, root):
        if not root: return
        self.inOrder(root.left)
        #print(self.ix)
        root.val += self.sums[self.ix]
        self.ix += 1
        self.inOrder(root.right)
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.ix = 0
        self.sums = []
        self.vals = []
        self.in_(root)
        sum = 0
        #print(self.vals)
        for i in self.vals[::-1]:
            self.sums = [sum] +self.sums
            sum += i
        #print(self.sums)
        self.inOrder(root)
        return root

        
def main():
    root = TreeNode(5)
    root.left = TreeNode(2)
    root.right = TreeNode(13)
    
    s = Solution().convertBST(root)
    inOrder(s)
    
if __name__=='__main__':
    main()