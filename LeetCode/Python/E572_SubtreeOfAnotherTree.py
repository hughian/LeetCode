class Solution:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        def isSame(s, t):
            if s==None and t==None: return True
            elif (not s and t) or (s and not t): return False
            else: return s.val==t.val and isSame(s.left, t.left) and isSame(s.right, t.right)
        def dfs(s, t):
            return isSame(s, t) or s and any((dfs(s.left, t),dfs(s.right, t))) #s is not None -> s.left, s.right
        return dfs(s,t)