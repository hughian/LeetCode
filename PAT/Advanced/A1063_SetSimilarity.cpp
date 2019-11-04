#include<iostream>
#include<vector>
#include<set>
#include<cstdio>
using namespace std;
int printf(const char*,...);
vector< set<int> > sets(52);
int N,M,K;
double check(int a,int b)
{
    set<int>::iterator it = sets[a].begin();
    double Nt=0.0,Nc=0.0;
    for(;it != sets[a].end();it++){
        if(sets[b].count(*it) == 1)
            Nt++;
    }
    Nc = sets[a].size() + sets[b].size() - Nt;
    return Nt/Nc * 100;
}
int main()
{
    int t;
    cin>>N;
    for(int i=1;i<=N;i++){
        cin>>M;
        for(int j=0;j<M;j++){
            cin>>t;
            sets[i].insert(t);
        }
    }
    cin>>K;
    int a,b;
    for(int i=0;i<K;i++){
        cin>>a>>b;
        printf("%.1lf\%%\n",check(a,b)); //打印百分号要使用 %%
    }
    return 0;
}
