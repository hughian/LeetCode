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
};
*/

// 题目二: 删除链表中重复节点
// 用 map（或者 unordered_map） 把节点值存一下，组后只保留只出现一次的节点
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if (pHead == nullptr)
            return nullptr;
        map<int, int> mp;
        ListNode *p = pHead, *_next;
        while(p){
            mp[p->val] ++;
            p = p->next;
        }

        ListNode *dummy = new ListNode(0);
        dummy->next = pHead;

        ListNode *t = dummy;
        p = pHead;

        while (p){
            if (mp[p->val] > 1){
                _next = p->next;
                t->next = p->next;
                p->next = nullptr;
                delete p;
                p = _next;
            }else{
                t = t->next;
                p = p->next;
            }
        }
        return dummy->next;
    }
};