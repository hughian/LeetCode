#include<iostream>
#include<vector>
#include<cstdio>
#include<algorithm>
using namespace std;
int printf(const char*,...);
struct Node{
    int data,next;
    Node():data(-1),next(-1){}
};

vector<Node> list(100000);
int head,N,K;

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
