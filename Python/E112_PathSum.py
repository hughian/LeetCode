# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
    def __repr__(self):
        return 'val:' + str(self.val) + ' ' + str(self.left) + ' ' + str(self.right)
        
class Solution(object):
    def preOrder(self, root, s):
        if not root:return
        if not root.left and not root.right:#leaf
            self.l += [s+root.val]
        else:
            self.preOrder(root.left,root.val+s)
            self.preOrder(root.right,root.val+s)
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        self.l = []
        self.preOrder(root,0)
        return sum in self.l
        
def main():
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.right= TreeNode(8)
    root.left.left = TreeNode(11)
    root.right.left= TreeNode(13)
    root.right.right=TreeNode(4)
    root.left.left.left=TreeNode(7)
    root.left.left.right=TreeNode(2)
    root.right.right.left=TreeNode(5)
    root.right.right.right=TreeNode(1)
    
    s = Solution().pathSum(root,22)
    print(s)
    
if __name__=='__main__':
    main()