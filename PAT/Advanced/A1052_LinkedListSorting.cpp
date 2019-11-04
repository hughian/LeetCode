#include<iostream>
#include<vector>
#include<algorithm>
#include<cstdio>
using namespace std;
int printf(const char*,...);
struct Node{
    int addr;
    int data;
    Node():addr(0),data(0){}
    Node(int a,int d):addr(a),data(d){}
    bool operator < (const Node &a)const{
        return this->data < a.data;
    }
};
vector<Node> vm(100010);
vector<Node> vn;
int N;
int main(){
    int head;
    cin>>N>>head;
    int addr;
    for(int i=0;i<N;i++){
        cin>>addr;
        cin>>vm[addr].data>>vm[addr].addr;
    }
    int p = head;
    while(p!=-1){
        vn.push_back(Node(p,vm[p].data));  
        p = vm[p].addr;
    }
    sort(vn.begin(),vn.end());
	if(vn.size()==0){
		printf("0 -1");  //空链表
		return 0;
	}
    printf("%d %05d\n",vn.size(),vn[0].addr);
    printf("%05d %d",vn[0].addr,vn[0].data);
    for(unsigned i=1;i<vn.size();i++){
        printf(" %05d\n%05d %d",vn[i].addr,vn[i].addr,vn[i].data);
    }
    printf(" -1\n");
    return 0;
}
