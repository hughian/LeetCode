# Definition for a binary tree node.
class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children
class Solution(object):
    def postorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        def post(root, t):
            if not root: return
            for c in root.children:
                post(c, t)
            t += [root.val]
        t = []
        post(root, t)
        return t
    def postorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        if not root: return []
        sta = [root]
        last = root
        res = []
        while(len(sta)):
            t = sta[-1]
            if (len(t.children)==0) or (last==t.children[-1]): #no children or all children have been visited
                sta.pop()
                res.append(t.val)
                last = t
            else:
                for c in t.children[::-1]:  #push into stack reversely
                    sta.append(c)
        return res


def main():
    n5 = Node(5,[])
    n6 = Node(6,[])
    n3 = Node(3,[n5,n6])
    n2 = Node(2,[])
    n4 = Node(4,[])
    root = Node(1,[n3,n2,n4])
    s = Solution().postorder(root)
    print(s)

if __name__ == '__main__':
    main()
    