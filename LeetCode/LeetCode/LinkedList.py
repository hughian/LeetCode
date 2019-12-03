
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


class DoublyListNode:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None


class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = ListNode(None)
        self.tail = self.head
        self.size = 0

    def _print_(self):
        p = self.head.next
        while p:
            print(p.val, end=' ')
            p = p.next
        print('')

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        print('get', index, self.size)
        self._print_()

        if index >= self.size:
            return -1
        i = -1
        p = self.head
        while p:
            p = p.next
            i += 1
            if i == index:
                return p.val
        return -1

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list.
        After the insertion, the new node will be the first node of the linked list.
        """
        p = ListNode(val)
        p.next = self.head.next
        self.head.next = p
        if self.tail == self.head:
            self.tail = p
        self.size += 1

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        print('tail=', self.tail.val)
        p = ListNode(val)
        p.next = self.tail.next
        self.tail.next = p
        self.tail = p
        self.size += 1
        self._print_()

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list.
        If index equals to the length of linked list, the node will be appended
        to the end of linked list. If index is greater than the length, the
        node will not be inserted.
        """
        if index == self.size:
            self.addAtTail(val)
        elif index < self.size:
            i = -1
            q = p = self.head
            while p:
                q = p
                p = p.next
                i += 1
                if i == index:
                    break
            if q:
                node = ListNode(val)
                node.next = q.next
                q.next = node
                if self.tail == self.head:
                    self.tail = node
                self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < self.size:
            i = -1
            q = p = self.head
            while p:
                q = p
                p = p.next
                i += 1
                if i == index:
                    break
            if q and p:
                if p == self.tail:
                    self.tail = q
                q.next = p.next
                p.next = None
                del p
                self.size -= 1


class MyDoublyList:
    # used for LRU cache
    def __init__(self):
        self.head = DoublyListNode(None, None)
        self.head.next = self.head
        self.head.prev = self.head
        self.size = 0

    def get(self, pt):
        return pt.key, pt.val

    def set(self, pt, key=None, val=None):
        if key:
            pt.key = key
        if val:
            pt.val = val

    def is_tail(self, pt):
        return pt == self.head.prev and pt != self.head

    def pop_head(self):
        if self.head.next != self.head:
            return self.remove(self.head.next)
        return None

    def remove(self, pt):
        q = pt.prev
        t = pt.next
        q.next = t
        t.prev = q

        pt.next = None
        pt.prev = None
        self.size -= 1
        return pt

    def insert(self, pt):
        pt.next = self.head
        pt.prev = self.head.prev
        self.head.prev = pt
        pt.prev.next = pt
        self.size += 1
        return pt

    def print(self):
        p = self.head.next
        while p != self.head:
            print("[%s:%s]->" % (p.key, p.val), end='')
            p = p.next
        print('Null')


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = MyDoublyList()
        self.key_to_ptr = {}

    def get(self, key: int) -> int:
        p = self.key_to_ptr.get(key, None)
        if not p:
            return -1
        if not self.cache.is_tail(p):
            p = self.cache.remove(p)
            self.cache.insert(p)
        return p.val

    def put(self, key: int, value: int) -> None:
        p = self.key_to_ptr.get(key, None)
        if p:
            if not self.cache.is_tail(p):
                p = self.cache.remove(p)
                self.cache.insert(p)
            self.cache.set(p, key=p.key, val=value)
        else:
            if self.cache.size >= self.capacity:
                pt = self.cache.pop_head()
                self.key_to_ptr.pop(pt.key)
                if pt:
                    del pt
            p = DoublyListNode(key, v=value)
            self.cache.insert(p)
            self.key_to_ptr[key] = p


class Solution_92:
    def reverseBetween(self, head, m, n):

        if m == n:
            return head
        new_head = ListNode(None)
        new_head.next = head
        prev = q = p = new_head
        cnt = 0
        while p and cnt <= n:

            if cnt < m:
                prev = q = p
                p = p.next
            elif cnt <= n:
                t = p.next
                p.next = q
                q = p
                p = t
            cnt += 1

        prev.next.next = p
        prev.next = q

        return new_head.next


class Solution_147:
    # 链表插入排序

    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return head

        dummy = prev = ListNode(float('-inf'))
        dummy.next = head

        while head and head.next:
            if head.val <= head.next.val:
                head = head.next
            else:
                p = head.next
                t = head.next.next
                head.next = t  # 把head.next从链上取下
                while p.val > prev.next.val:
                    prev = prev.next

                next_prev = prev.next
                prev.next = p
                p.next = next_prev
                prev = dummy

        return dummy.next


class Solution_25:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        pass
