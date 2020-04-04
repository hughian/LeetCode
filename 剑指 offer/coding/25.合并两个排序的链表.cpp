//
// Created by Hughian on 2020/4/4.
//


struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};

class Solution {
public:
    // 新建一条链表，使用尾插法
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode* node = new ListNode(0);
        ListNode* dummy = node;
        while (pHead1 && pHead2){
            if (pHead1->val <= pHead2->val){
                node->next = pHead1;
                pHead1 = pHead1 -> next;
            }else{
                node->next = pHead2;
                pHead2 = pHead2 -> next;
            }
            node = node->next;
        }
        node->next = nullptr;
        if (pHead1){
            node->next = pHead1;
        }
        if (pHead2){
            node->next = pHead2;
        }
        return dummy->next;
    }
};