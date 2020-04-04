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
    // 三指针
    ListNode* ReverseList(ListNode* pHead) {
        if (pHead == nullptr || pHead->next == nullptr)  // edge case
            return pHead;
        ListNode *cur = pHead->next, *prev=pHead, *next;
        while(cur){
            next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        pHead->next = nullptr;
        return prev;
    }

    // 使用头插法
    ListNode* ReverseList(ListNode* pHead) {
        if (pHead == nullptr || pHead->next == nullptr)  // edge case
            return pHead;
        ListNode *dummy = new ListNode(-1), *cur=pHead, *next;
        while (cur){
            next = cur->next;
            // cur->next = nullptr;
            // 头插
            cur->next = dummy->next;
            dummy->next = cur;

            cur = next;
        }

        return dummy->next;
    }
};