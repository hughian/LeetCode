# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    
    def bst(self, preorder):
        if len(preorder) == 0:
            return None
        inorder = sorted(preorder)
        root = TreeNode(preorder[0])
        r = inorder.index(preorder[0])
        root.left = self.bst(preorder[1:r+1])
        root.right = self.bst(preorder[r+1:])
        return root
    
    def bstFromPreorder(self, preorder) -> TreeNode:
        return self.bst(preorder)
        '''
        root = TreeNode(preorder[0])
        
        for x in preorder[1:]:
            p, q= root, root
            while p:
                q = p
                if x < p.val:
                    p = p.left
                else:
                    p = p.right
                    
            if x < q.val:
                q.left = TreeNode(x)
            else:
                q.right = TreeNode(x)
        return root
        '''

def preOrder(root):
    if root:
        print(root.val, end=" ")
        preOrder(root.left)
        preOrder(root.right)

s = Solution()
root = s.bst([8,5,1,7,10, 12])
preOrder(root)
