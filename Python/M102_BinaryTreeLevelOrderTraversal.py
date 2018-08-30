# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return [] # if root==None return an empty list
        ans,tmp = [], []
        que = [root,None]
        while len(que)>1:
            t = que[0]
            que = que[1:]
            if t==None:
                ans = ans + [tmp]
                tmp = []
                que += [None]
            else:
                tmp += [t.val]
                if t.left : que += [t.left]
                if t.right: que += [t.right]
        ans = ans + [tmp]
        return ans

def main():
    root = TreeNode(3)
    root.left=TreeNode(9)
    root.right=TreeNode(20)
    root.right.left=TreeNode(15)
    root.right.right=TreeNode(7)
    
    ll = Solution().levelOrderBottom(None)
    print(ll)
    
if __name__=='__main__':
    main()