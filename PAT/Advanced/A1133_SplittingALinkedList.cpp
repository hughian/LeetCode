#include<iostream>
#include<cstdio>
#include<vector>
#include<map> //不能使用map映射数据到地址，数据可能相同
using namespace std;
struct Node{
    int data;
    int next;
    Node():data(0),next(0){}
};

void print(vector<Node>& vec,int head){
    int p = head;
    while(p!=-1){
        cout<<vec[p].data<<" ";
        p = vec[p].next;
    }
}

void printv(vector<int> &v,vector<Node>& vec)
{
    printf("%05d %d ",v[0],vec[v[0]].data);
    for(int i=1;i<(int)v.size();i++){
        printf("%05d\n%05d %d ",v[i],v[i],vec[v[i]].data);
    }
    printf("-1");
}

void copy(vector<int>&des,vector<int>&src)
{
    for(int i=0;i<(int)src.size();i++)
        des.push_back(src[i]);
}

int main()
{
    int head,N,K;
    vector<Node> vec(100001,Node());
    cin>>head>>N>>K;
    int addr,data,next;
    for(int i=0;i<N;i++){
        cin>>addr>>data>>next;
        vec[addr].data = data;
        vec[addr].next = next;
    }
    
    vector<int> neg;
    vector<int> zerok;
    vector<int> greater;
    
    int p = head;
    while(p!=-1){
        int d = vec[p].data;
        if(d<0)
            neg.push_back(p);
        else if(d >=0 && d<=K)
            zerok.push_back(p);
        else
            greater.push_back(p);
        p = vec[p].next;
    }

    copy(neg,zerok);
    copy(neg,greater);
    printv(neg,vec);
	return 0;
}
