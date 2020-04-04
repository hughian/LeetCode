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

// 使用 helper func 来完成，返回前将结果输出。
class Solution {
public:
    void helper(ListNode *p, vector<int> & vec){
        if (p != nullptr){
            helper(p->next, vec);
            vec.emplace_back(p->val);
        }
    }
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> res;
        helper(head, res);
        return res;
    }
};