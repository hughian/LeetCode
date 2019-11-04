#include<iostream>
#include<vector>
#include<stack>
#include<cstdio>
using namespace std;
int printf(const char*,...);
struct Node{
    char data;
    int next;
    Node():data(0),next(0){}
};
vector<Node>  link(100000);

int main()
{
    int head1,head2,N,addr;
    cin>>head1>>head2>>N;
    for(int i=0;i<N;i++){
        cin>>addr;
        cin>>link[addr].data>>link[addr].next;
    }
    stack<int> s1,s2;
    int p = head1;
    while(p!=-1){
        s1.push(p);
        p = link[p].next;
    }
    p = head2;
    while(p!=-1){
        s2.push(p);
        p = link[p].next;
    }
    int ans=-1;
    while(!s1.empty() && !s2.empty()){
        if(s1.top() == s2.top()){
			ans = s1.top();
            s1.pop(); s2.pop();
        }else
            break;
    }
    if(ans == -1)
        printf("-1");
    else 
        printf("%05d",ans);
    return 0;
}
