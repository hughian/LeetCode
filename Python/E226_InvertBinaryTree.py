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
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:return
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root

        
def main():
    root = TreeNode(4)
    root.left = TreeNode(2)
    root.right= TreeNode(7)
    root.left.left = TreeNode(1)
    root.left.right= TreeNode(3)
    root.right.left= TreeNode(6)
    root.right.right=TreeNode(9)
    
    root = Solution().invertTree(root)
    preOrder(root)
    
if __name__=='__main__':
    main()