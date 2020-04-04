//
// Created by Hughian on 2020/4/4.
//

/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode *FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2) {
        int m = 0, n = 0;
        ListNode *p=nullptr, *t=nullptr;
        // 数两条链表的节点数目
        for(p = pHead1; p; m++, p=p->next);
        for(t = pHead2; t; n++, t=t->next);
        // 这两个循环只会有一个被执行
        for (p=pHead1; m > n; p = p->next, m--);
        for (t=pHead2; n > m; t = t->next, n--);

        while (p && t) {
            if (t == p)
                return t;
            t = t->next;
            p = p->next;
        }
        return nullptr;
    }
};