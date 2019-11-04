#Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def height(self,node):
        hleft = 0 if not node.left else self.height(node.left)
        hright = 0 if not node.right else self.height(node.right)
        if hleft + hright > self.m:
            self.m = hleft + hright
        return max(hleft,hright) + 1
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.m = 0
        if not root: return 0
        self.height(root)
        return self.m
    def diameterOfBinaryTree_(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.m = 0
        
        def height_(node):
            hleft = 0 if node.left==None else height_(node.left)
            hright = 0 if node.right==None else height_(node.right)
            if hleft + hright > self.m:
                self.m = hleft + hright
            return max(hleft, hright) + 1
        
        if not root: return 0
        height_(root)
        return self.m

def main():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    ret = Solution().diameterOfBinaryTree_(root)
    out = str(ret)
    print(out)


if __name__ == '__main__':
    main()