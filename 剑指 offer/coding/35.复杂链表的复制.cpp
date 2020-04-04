//
// Created by Hughian on 2020/4/4.
//

/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/

// 使用 hash, 链表节点前后可以作为 hash 关系
// 把复制出来的节点放到原来节点的后面
class Solution {
public:
    RandomListNode *Clone(RandomListNode *pHead) {
        if (pHead == nullptr)
            return nullptr;

        RandomListNode *cur = nullptr, *node;
        RandomListNode *t;
        // 复制节点
        cur = pHead;
        while (cur) {
            auto *node = new RandomListNode(cur->label);
            node->next = cur->next;
            cur->next = node;
            cur = node->next;
        }
        // 复制 random 指针的指向
        cur = pHead;
        while (cur) {
            if (cur->random)
                cur->next->random = cur->random->next;
            cur = cur->next->next;
        }
        // 把复制的节点摘下啦组成复制的链表
        cur = pHead;
        t = pHead->next;
        while (cur) {
            node = cur->next;
            cur->next = node->next;
            if (cur->next)
                node->next = cur->next->next;
            else
                node->next = nullptr;
            cur = cur->next;
        }

        return t;
    }
};
