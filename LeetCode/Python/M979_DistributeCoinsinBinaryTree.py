class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def distributeCoins(self, root: TreeNode) -> int:
        """
        move means how many current node can give to its parent. which is node.val - 1
        """
        move = 0
        
        def preOrder(root):
            nonlocal move
            if not root:
                return 0
            e = preOrder(root.left)
            r = preOrder(root.right)
            
            move += abs(root.val + e + r - 1)
            print(abs(root.val + e + r - 1))
            return root.val + e + r - 1
        
        preOrder(root)
        return move
        
root = TreeNode(1)
root.left = TreeNode(0)
root.right = TreeNode(0)
root.left.left = TreeNode(3)

s = Solution()
r = s.distributeCoins(root)
print(r)
