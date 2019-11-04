#include<iostream>
#include<string>
#include<vector>
#include<cstdio>
using namespace std;
string award[4]={"Mystery Award","Minion","Chocolate","Checked"};
int isPrime(int x){
    for(int i=2;i<x;i++){
        if(x%i==0)
            return 2;
    }
    return 1;
}
int main()
{
    int n,id;
    vector<int> rank(10001,-1);
    cin>>n>>id;
    rank[id] = 0;
    for(int i=2;i<=n;i++){
        cin>>id;
        rank[id] = isPrime(i);
    }
    int k;
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>id;
        if(rank[id]==-1)
            printf("%04d: Are you kidding?\n",id);
        else{
            printf("%04d: %s\n",id,award[rank[id]].c_str());
            rank[id] = 3;
        }
    }
    return 0;
}
