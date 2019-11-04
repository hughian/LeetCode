class Solution{
public:
    ListNode * addTwoNumbers(ListNode* l1,ListNode* l2){
        char s,a,b,c=0;
        ListNode *p1=l1,*p2=l2;
        ListNode *phead=new ListNode(-1);
        ListNode *head = phead;
        ListNode *pa;
        while(p1 && p2){
            a = p1->val;
            b = p2->val;
            s = a + b + c;
            c = (s > 9);
            s = s % 10;
            pa = new ListNode(s);
            pa->next = head->next;
            head->next = pa;
            head = pa;
            p1 = p1->next;
            p2 = p2->next;
        }
        if(!p1){
            while(p2){
                b = p2->val;
                s = b + c;
                c = (s > 9);
                s = s%10;
                pa = new ListNode(s);
                pa->next = head->next;
                head->next = pa;
                head =pa;
                p2 = p2->next;
            }
        }
        else{
            while(p1){
                a = p1->val;
                s = a + c;
                c = (s > 9);
                s = s % 10;
                pa = new ListNode(s);
                pa->next = head->next;
                head->next = pa;
                head = pa;
                p1 = p1->next;
            }
        }

        if(c){
            pa = new ListNode(c);
            pa->next = head->next;
            head->next = pa;
            head=pa;
        }
        return phead->next;
    }
};
