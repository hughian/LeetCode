# Definition for a binary tree node.
from copy import deepcopy
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def allPossibleFBT(self, N: int):
        if N % 2 == 0:
            return []
        if N == 1:
            return [TreeNode(0)]
        
        memo = {0:[], 1:[TreeNode(0)]}
        
        def helper(N):
            if N not in memo:
                memo[N] = []
                for x in range(N):
                    y = N - 1 - x
                    for left in helper(x):
                        for right in helper(y):
                            root = TreeNode(0)
                            root.left = left
                            root.right = right
                            memo[N] += [root]
            return memo[N]                            
        helper(N)
        return memo[N]

def level(root):
    que = [root]
    ans = []
    while que:
        t = que.pop(0)
        if t: ans.append(t.val)
        else:
            ans.append(None)
            continue
        que.append(t.left)
        que.append(t.right)
    return ans

s = Solution()
r = s.allPossibleFBT(9)
r = set(r)
print(r, '\n', len(r))
for t in r:
    ans = level(t)
    print(ans.count(0))


