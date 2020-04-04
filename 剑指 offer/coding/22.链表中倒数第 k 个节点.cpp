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

// 两边遍历的解法
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        int n=0;

        ListNode *p = pListHead;
        while (p){
            n++;
            p = p->next;
        }
        if (n < k)
            return nullptr;

        p = pListHead;
        int cnt = 0;
        while(p){

            if (cnt == n-k)
                break;
            p = p->next;
            cnt += 1;
        }

        return p;
    }
};