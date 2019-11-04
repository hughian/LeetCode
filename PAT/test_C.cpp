#include<iostream>
#include<vector>
#include<stack>
using namespace std;
vector< vector<int> > arc(1010,vector<int>(1010,0));
vector<bool> used(1010,false);
vector<int> inde(1010,false);
int N,M;
stack<int> s;
vector<int> ans;
int Max = 0;
void dfs(int v){
    used[v] = true;
    for(int i=0;i<N;i++){
        if(!used[i] && arc[v][i]==1){
            arc[v][i] = 0;
            dfs(i);
        }
    }
    s.push(v);
}

    vector<bool> vis(1010,false);
int getinde()
{
    int tmp;
    int v = -1;
    for(int j=0;j<N;j++){
        tmp = 0;
        for(int i=0;i<N;i++){
            tmp += arc[i][j];
        }
        if(tmp==0 && v==-1 && !vis[j]){
            v = j;
            vis[j] = true;
        }
        inde[j] = tmp;
    }
    return v;
}

int main()
{
    cin>>N>>M;
    int a,b;
    for(int i=0;i<M;i++){
        cin>>a>>b;
        arc[a][b] = 1;
    }
    int t;
    while((t=getinde())!=-1){
            ans.push_back(t);
            for(int i=0;i<N;i++){
                arc[t][i] = 0;
            }
    }
    if((int)ans.size()==N){
        cout<<"YES"<<endl;
        for(int i=0;i<N;i++){
            cout<<ans[i];
            if(i<N-1) cout<<" ";
        }
    }else{
        cout<<"NO"<<endl;
        cout<<N-ans.size()<<endl;
    }
    return 0;
}
