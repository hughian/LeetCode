#include<iostream>
#include<stack>
using namespace std;
struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x):val(x),next(0){}
};
class Solution{
public:
    ListNode * addTwoNumbers(ListNode* l1,ListNode* l2){
        stack<char> s1,s2;
        char s,a,b,c=0;
        ListNode *p1=l1,*p2=l2;
        ListNode *head=new ListNode(-1);
        ListNode *pa;
        while(p1){
            s1.push(p1->val);
            p1=p1->next;
        }
        while(p2){
            s2.push(p2->val);
            p2=p2->next;
        }
        
        while(!s1.empty() && !s2.empty()){
            b=s2.top();
            s2.pop();
            a=s1.top();
            s1.pop();
            s=a+b+c;
            c=s>9?1:0;
            s=c?s-10:s;
            pa=new ListNode(s);
            pa->next=head->next;
            head->next=pa;
        }
        if(s2.empty()){
            while(!s1.empty()){
                a=s1.top();
                s1.pop();
                s=a+c;
                c=s>9?1:0;
                s=c?(s-10):s;
                pa=new ListNode(s);
                pa->next=head->next;
                head->next=pa;
            }
        }
        else{
            while(!s2.empty()){
                a=s2.top();
                s2.pop();
                s=a+c;
                c=s>9?1:0;
                s=c?(s-10):s;
                pa=new ListNode(s);
                pa->next=head->next;
                head->next=pa;
            }
        }
        if(c){
            pa=new ListNode(c);
            pa->next=head->next;
            head->next=pa;
        }
        return head->next;
    }
};


int main()
{
    ListNode *ph1=new ListNode(9);
    ListNode *ph2=new ListNode(9);
    ListNode *tmp;
    int i=7;
    while(i<9){
        tmp=new ListNode(i++);
        tmp->next=ph1->next;
        ph1->next=tmp;
    }
    i=8;
    while(i<9){
        tmp=new ListNode(i++);
        tmp->next=ph2->next;
        ph2->next=tmp;
    }
    Solution a;
    tmp=a.addTwoNumbers(ph1,ph2);
    while(tmp){
        cout<<tmp->val<<" ";
        tmp=tmp->next;
    }
    return 0;
}
