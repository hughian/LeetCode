#include<iostream>
#include<map>
#include<vector>
#include<cstdio>
using namespace std;

struct Node{
    int data;
    int next;
    Node(int d,int n):data(d),next(n){}
};

vector<Node> vn(100000,Node(-1,-1));
map<int,bool> used;

int abss(int x){
    return x>=0?x:(-x);
}

int main()
{
    int addr,key,next,head,n;
    cin>>head>>n;
    for(int i=0;i<n;i++){
        cin>>addr>>key>>next;
        vn[addr].data = key;
        vn[addr].next = next;
    }
    int p = head,pre=head;
    int rmhead=-1,rmtail;
    
    int a = abss(vn[p].data);
    used[a] = true;
    p = vn[p].next;
    while(p!=-1){
        a = abss(vn[p].data);
        if(used[a]==true){
            int q = p;
            p = vn[p].next;
            vn[pre].next = p;
            vn[q].next = -1;
            if(rmhead==-1){
                rmhead = q;
                rmtail = q;
            }else{
                vn[rmtail].next = q;
                rmtail = q;
            }
        }else{
            used[a] = true;
            p = vn[p].next;
            pre = vn[pre].next;
        }
    }
    p = head;
    printf("%05d %d",p,vn[p].data);
    p = vn[p].next;
    while(p!=-1){
        printf(" %05d\n%05d %d",p,p,vn[p].data);
        p = vn[p].next;
    }
    printf(" -1\n");
    if(rmhead !=- 1){
        p = rmhead;
        printf("%05d %d",p,vn[p].data);
        p = vn[p].next;
        while(p!=-1){
            printf(" %05d\n%05d %d",p,p,vn[p].data);
            p= vn[p].next;
        }
        printf(" -1\n");
    }
    return 0;
}
