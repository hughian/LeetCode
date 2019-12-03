import collections
from typing import List

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __repr__(self):
        return f'{{ {self.val}, left:{{{self.left}}}, right{{{self.right}}}}}'


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

    def __repr__(self):
        if self.next is None:
            return f'{self.val}'
        else:
            return f'{self.val} -> {self.next}'

        # return f'{self.val}, {{next:{self.next}}}'


class Tree:
    def __init__(self, root):
        self.root = root

    def pre_recursive(self):
        res = []

        def pre(root):
            nonlocal res
            if root:
                res += [root.val]
                pre(root.left)
                pre(root.right)

        pre(self.root)
        return res

    def pre_non_recursive(self):
        stack = []
        res = []
        p = self.root
        while stack or p:
            while p:
                res += [p.val]
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                p = p.right
        return res

    def in_recursive(self):
        res = []

        def in_order(root):
            nonlocal res
            if root:
                in_order(root.left)
                res += [root.val]
                in_order(root.right)

        in_order(self.root)
        return res

    def in_non_recursive(self):
        stack = []
        res = []
        p = self.root
        while stack or p:
            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                res += [p.val]
                p = p.right
        return res

    def post_recursive(self):
        res = []

        def post(root):
            nonlocal res
            if root:
                post(root.left)
                post(root.right)
                res += [root.val]

        post(self.root)
        return res

    def post_non_recursive(self):
        res = []
        stack = []
        p = self.root
        last = None
        while stack or p:

            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack[-1]
                if p.right is None or last == p.right:
                    res += [p.val]
                    last = p
                    stack.pop()
                    p = None
                else:
                    p = p.right
        return res

    @staticmethod
    def run():
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        root.right.left = TreeNode(5)
        root.right.right = TreeNode(7)
        tree = Tree(root)
        print('Pre order:')
        print(tree.pre_recursive())
        print(tree.pre_non_recursive())
        print('In order:')
        print(tree.in_recursive())
        print(tree.in_non_recursive())
        print('Post order:')
        print(tree.post_recursive())
        print(tree.post_non_recursive())


class Solution_235:
    def _0_lowestCommonAncestor(self, root, p, q):
        """
        using bool flags to track whether we have encountered the nodes(p and q).
        if we have travel through both nodes, say, two of the flags was set True,
        then we find our answer.

        Time complexity: O(n)
        Space complexity: O(n)
        """
        ans = None

        def post(root):
            nonlocal ans
            if not root: return False
            left = post(root.left)
            right = post(root.right)
            mid = root == p or root == q
            if mid + left + right >= 2:
                ans = root
            return mid or left or right

        post(root)
        return ans

    def _1_lowestCommonAncestor(self, root, p, q):
        """using parent pointer"""
        parent = {root: None}
        stack = [root]
        while p not in parent or q not in parent:
            t = stack.pop()
            if t.left:
                parent[t.left] = t
                stack.append(t.left)
            if t.right:
                parent[t.right] = t
                stack.append(t.right)
        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]

        while q not in ancestors:
            q = parent[q]
        return q

    def _2_lowestCommonAncestor(self, root, p, q):
        stack = [(root, 0)]  # 0: unsearched, 1: left searched, 2: both searched
        one_found = False
        LCA_index = -1
        while stack:
            t, t_state = stack[-1]
            if t_state == 0:
                if t == p or t == q:
                    if one_found:
                        return stack[LCA_index][0]
                    else:
                        one_found = True
                        LCA_index = len(stack) - 1
                stack.pop()
                stack.append((t, t_state + 1))
                if t.left:
                    stack.append((t.left, 0))
            elif t_state == 1:
                stack.pop()
                stack.append((t, t_state + 1))
                if t.right:
                    stack.append((t.right, 0))
            else:
                if one_found and LCA_index == len(stack) - 1:
                    LCA_index -= 1
                stack.pop()
        return None

    def _3_lowestCommonAncestor(self, root, p, q):
        def foo(root):
            if p.val > root.val and q.val > root.val:  # 这里决定了一定有右子树
                return foo(root.right)
            elif p.val < root.val and q.val < root.val:  # 这里决定了一定有左子树
                return foo(root.left)
            else:
                return root

        return foo(root)

    def _lowestCommonAncestor(self, root, p, q):
        t = root
        while t:
            if p.val > t.val and q.val > t.val:
                t = t.right
            elif p.val < t.val and q.val < t.val:
                t = t.left
            else:
                return t

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > q.val:
            p, q = q, p

        def post(root):
            if not root: return None
            rleft = post(root.left)
            rright = post(root.right)
            if rleft or rright:
                if p.val <= root.val <= q.val:
                    return root
                else:
                    return rleft or rright
            else:
                if p.val <= root.val <= q.val:
                    return root
                else:
                    return None

        return post(root)

class Solution_95:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def gen_tree(lo, hi):
            if lo > hi:
                return [None]
            res = []
            for i in range(lo, hi + 1):
                left_ls = gen_tree(lo, i - 1)
                right_ls = gen_tree(i + 1, hi)
                for left in left_ls:
                    for right in right_ls:
                        root = TreeNode(i)
                        root.left = left
                        root.right = right
                        res.append(root)
            return res

        if n < 1:
            return []
        return gen_tree(1, n)

class Solution_109:
    def _sortedListToBST(self, head: ListNode) -> TreeNode:
        ls = []
        p = head
        # 把节点存在数组里，也可以用快慢指针来找中点
        while p:
            ls.append(p.val)
            p = p.next

        def build_tree(inorder):
            if len(inorder) == 0:
                return None
            mid = len(inorder) // 2
            node = TreeNode(inorder[mid])
            node.left = build_tree(inorder[:mid])
            node.right = build_tree(inorder[mid + 1:])
            return node

        return build_tree(ls)

    def sortedListToBST(self, head: ListNode) -> TreeNode:
        # solution 3给出的方法太牛皮了
        n = 0
        p = head
        while p:
            n += 1
            p = p.next

        def convert(lo, hi):
            nonlocal head
            if lo > hi:
                return None

            mid = (lo + hi) >> 1
            left = convert(lo, mid - 1)  # 先向左递归

            node = TreeNode(head.val)
            node.left = left

            head = head.next

            node.right = convert(mid + 1, hi)
            return node

        return convert(0, n - 1)

    @staticmethod
    def run():
        head = ListNode(-10)
        head.next = ListNode(-5)
        head.next.next = ListNode(0)
        head.next.next.next = ListNode(5)
        head.next.next.next.next = ListNode(10)

        print(Solution_109().sortedListToBST(head))


class Codec_449:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        pre = []
        ino = []

        def pre_order(root):
            nonlocal pre
            if root:
                pre.append(root.val)
                pre_order(root.left)
                pre_order(root.right)

        def in_order(root):
            nonlocal ino
            if root:
                in_order(root.left)
                ino.append(root.val)
                in_order(root.right)

        pre_order(root)
        in_order(root)
        # print(pre)
        # print(ino)
        return str(pre) + '##' + str(ino)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        pre, ino = data.split('##')
        if len(pre) <= 2:
            return None
        pre = [int(x) for x in pre[1:-1].split(',')]
        ino = [int(x) for x in ino[1:-1].split(',')]

        # print(pre)
        # print(ino)
        def build(pre, ino):
            if len(pre) == 0:
                return None
            root = TreeNode(pre[0])
            idx = ino.index(pre[0])
            n = idx
            root.left = build(pre[1:n + 1], ino[:idx])
            root.right = build(pre[n + 1:], ino[idx + 1:])
            return root

        return build(pre, ino)
    #################################################################
    # another solution
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        def level(root):
            que = collections.deque([root])
            res = []
            while que:
                t = que.popleft()
                if t:
                    res.append(t.val)
                    que.append(t.left)
                    que.append(t.right)
                else:
                    res.append(None)

            while res and res[-1] == None:
                res.pop()
            return res

        lv = level(root)

        # print(lv)
        return str(lv)[1:-1]

    def deserialize(self, data):
        if len(data) == 0:
            return None
        data = data.replace(' ', '')
        nodes = [None if x == 'None' else TreeNode(int(x))
                 for x in data.split(',')]
        kids = nodes[::-1]
        root = kids.pop()
        for node in nodes:
            if node:
                if kids: node.left = kids.pop()
                if kids: node.right = kids.pop()
        return root

    def _deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if len(data) == 0:
            return None
        lv = [int(x) if x != ' None' else None for x in data.split(',')]
        # print(lv)

        que = collections.deque()
        root = TreeNode(lv[0])
        que.append(root)
        idx = 1
        while que:
            t = que.popleft()
            if idx < len(lv):
                if lv[idx] == None:
                    t.left = None
                else:
                    t.left = TreeNode(lv[idx])
                    que.append(t.left)
            if idx + 1 < len(lv):
                if lv[idx + 1] == None:
                    t.right = None
                else:
                    t.right = TreeNode(lv[idx + 1])
                    que.append(t.right)
            idx += 2

        return root


#####################################################################################################################
# Trie 前缀树
class TrieNode:
    def __init__(self):
        self.child = [None] * 26
        self.end_of_word = False


class Trie_208:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        p = self.root
        for c in word:
            ix = ord(c.lower()) - ord('a')
            if p.child[ix] is None:
                p.child[ix] = TrieNode()
            p = p.child[ix]
        p.end_of_word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        p = self.root
        for c in word:
            ix = ord(c.lower()) - ord('a')
            if p.child[ix] is None:
                return False
            p = p.child[ix]
        return p.end_of_word

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        p = self.root
        for c in prefix:
            ix = ord(c.lower()) - ord('a')
            if p.child[ix] is None:
                return False
            p = p.child[ix]
        return True
