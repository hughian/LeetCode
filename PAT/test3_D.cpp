#include<iostream>
#include<vector>
#include<map>
using namespace std;
vector< vector<pair<int,int> > > vec(1e6+10);
vector<int> vp;
map<int,bool> mpk;
vector<bool> used(1e6+10,false);
int N,M,P,K;
void dfs(int v,int len){
    used[v] = true;
    for(unsigned i=0;i<vec[v].size();i++){
        int ix = vec[v][i].first;
        int w = vec[v][i].second;
        if(!used[ix]){
            dfs(ix,len+w);
        }
    }
    used[v] = false;
}
int main()
{
    int a,b,t;
    cin>>N>>M>>P>>K;
    for(int i=0;i<P;i++){
        cin>>t;
        vp.push_back(t);
    }
    for(int i=0;i<K;i++){
        cin>>t;
        mpk[t] = true;
    }
    for(int i=0;i<M;i++){
        cin>>a>>b>>t;
        vec[a].push_back(make_pair(b,t));
    }

    for(int i=0;i<P;i++){
        dfs(vp[i],0);
    }
}
