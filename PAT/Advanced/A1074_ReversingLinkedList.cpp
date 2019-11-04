#include<iostream>
#include<vector>
#include<cstdio>
#include<algorithm>
using namespace std;
int printf(const char*,...);
struct Node{
    int data,next;
    Node(int d,int n):data(d),next(n){}
    Node():data(-1),next(-1){}
};

vector<Node> list(100000);
int head,N,K;

/*直接按照题意翻转，对链表的操作要求高，未能实现，下面函数可用*/
/*int reversePart(int phead,int ptail,int &ftail){
    int pre = phead;
    int p = list[phead].next;
    int q;
    while(p!=ptail){
        q = list[p].next;
        list[p].next = pre;
        pre = p;
        p = q;
    }
    list[ptail].next = pre;
    list[phead].next = ftail;
    ftail = ptail;
}*/

void print(int head)
{
    int p = head;
    while(p!=-1){
        printf("%05d %d %05d\n",p,list[p].data,list[p].next);
        p = list[p].next;
    }
}
int main()
{   
    cin>>head>>N>>K;
    int addr,data,next;
    for(int i=0;i<N;i++){
        cin>>addr>>data>>next;
        list[addr].data = data;
        list[addr].next = next;
    }
    vector<int> vn;
    int p = head;
    while(p!=-1){
        vn.push_back(p);
        p = list[p].next;
    }
    
	/*改用翻转地址的办法*/
    for(int i=0;i<(int)vn.size();i+=K){
        if(i+K<=(int)vn.size())
            reverse(vn.begin()+i,vn.begin()+i+K);
    }

    printf("%05d %d",vn[0],list[vn[0]].data);
    for(int i=1;i<(int)vn.size();i++){
        printf(" %05d\n%05d %d",vn[i],vn[i],list[vn[i]].data);
    }
    printf(" -1\n");

    return 0;
}
