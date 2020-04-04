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

// 快慢指针找有没有环，然后慢指针和新指针一起，相遇时就是环入口
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if (pHead == nullptr)
            return nullptr;
        ListNode *slow=pHead, *quick=pHead;
        while(slow && quick){
            slow = slow->next;
            quick = quick->next;
            if(quick)
                quick = quick->next;
            if(slow == quick)
                break;
        }
        if(slow == nullptr || quick == nullptr)
            return nullptr;

        ListNode *p = pHead;

        while(p != slow){
            p = p->next;
            slow = slow->next;
        }
        return p;
    }
};