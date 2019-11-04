#include<iostream>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
vector< vector<int> > edge(10010);
vector<bool> used(10010,false);
vector<int> ans;
int N;
void dfs(int v){
    used[v] = true;
    int len = edge[v].size();
    for(int i=0;i<len;i++){
        if(!used[edge[v][i]]){
            dfs(edge[v][i]);
        }
    }
}
int bfs(int v){
    int depth = 0;
    queue<int> q;
    used[v] = true;
    q.push(v);
    q.push(0);
    while(q.size()>1){
        int t = q.front();q.pop();
        if(t==0){
            depth++;
            q.push(0);
        }else{
            for(int i=0;i<(int)edge[t].size();i++){
                int idx = edge[t][i];
                if(!used[idx]){
                    used[idx] = true;
                    q.push(idx);
                }
            }
        }
    }
    return depth;
}
int main()
{
    cin>>N;
    int a,b;
    if(N==1){
        cout<<1;
        return 0;
    }
    for(int i=1;i<N;i++){
        cin>>a>>b;
        edge[a].push_back(b);
        edge[b].push_back(a);
    }
    int cnt = 0;
    for(int i=1;i<=N;i++){
        if(!used[i]){
            dfs(i);cnt++;
        }
    }
    if(cnt==1){
        vector<int> tpv,t;
        int max = 0,d;
        for(int i=1;i<=N;i++)
            if(edge[i].size()==1) tpv.push_back(i);
        for(int i=0;i<(int)tpv.size();i++){
			fill(used.begin(),used.end(),false);
            d = bfs(tpv[i]);
            t.push_back(d);
            if(d > max)
                max = d;
        }
        for(int i=0;i<(int)t.size();i++){
            if(t[i]==max)
                ans.push_back(tpv[i]);
        }
        sort(ans.begin(),ans.end());
        for(int i=0;i<(int)ans.size();i++)
            cout<<ans[i]<<endl;
    }else{
        cout<<"Error: "<<cnt<<" components";
    }
    return 0;
}
