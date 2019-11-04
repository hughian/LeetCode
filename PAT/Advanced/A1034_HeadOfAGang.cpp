#include<iostream>
#include<vector>
#include<algorithm>
#include<cstdio>
#include<cstring>
using namespace std;
int scanf(const char*,...);
int printf(const char*,...);
int strcpy(char*,const char*);
struct Node{
    char buf[5];
    int time;
    Node(char *s,int t):time(t){
        strcpy(buf,s);
    }
};
vector<bool> used(18000,false);
vector< vector<Node> > vn(18000);
int getidx(char buf[]){
    int ans = 0;
    for(int i=0;i<3;i++)
        ans += ans * 26 + buf[i] - 'A';
    return ans;
}
void dfs(int x)
{

}

int main()
{
    int n,k;
    cin>>n>>k;
    char a[5],b[5];
    int t;
    for(int i=0;i<n;i++){
        scanf("%s%s%d",a,b,t);
        int ai =getidx(a);
        int bi =getidx(b);
        vn[ai].push_back(Node(b,t));
        vn[bi].push_back(Node(a,t));
    }
    for(int i=0;i<18000;i++){
        if(vn[i].size()!=0){
            dfs(i);
        }
    }
}
