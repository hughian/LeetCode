# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
r = {}
class Solution:
    def sumnode(self,node):
        sleft = 0 if node.left==None else self.sumnode(node.left)
        sright= 0 if node.right==None else self.sumnode(node.right)
        s = sleft + sright + node.val
        if s in r.keys():
            r[s] += 1
        else:
            r[s] = 1
        return s
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        global r
        r = {}
        if not root:
            return []
        self.sumnode(root)
        l = list(set(r.values()))
        l.sort()
        tmp = [k for k in r.keys() if r[k]==l[-1]]
        return tmp
        
if __name__=='__main__':
    tree = TreeNode(1)
    #tree.left=TreeNode(2)
    #tree.right=TreeNode(-5)
    a = Solution()
    t = a.findFrequentTreeSum(tree)
    print(t)