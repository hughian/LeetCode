#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
vector< vector<int> > tree(101);
vector< vector<int> > ans;
vector<int> weight(101);
int N,M,S;
vector<int> tmp;
void dfs(int root,int sum){
    if(sum==S && tree[root].size()==0){
        ans.push_back(tmp);
    }
    for(unsigned i=0;i<tree[root].size();i++){
        int idx = tree[root][i];
        tmp.push_back(weight[idx]);
        dfs(idx,sum+weight[idx]);
        tmp.pop_back();
    }
}
bool cmp(vector<int>&a,vector<int>&b){
    int i=0;
    for(;i<min(a.size(),b.size());i++){
        if(a[i]!=b[i])
            break;
    }
    return a[i] > b[i];
}
int main()
{
    cin>>N>>M>>S;
    for(int i=0;i<N;i++)
        cin>>weight[i];
    int id,k,t;
    for(int i=0;i<M;i++){
        cin>>id>>k;
        for(int j=0;j<k;j++){
            cin>>t;
            tree[id].push_back(t);
        }
    }
    tmp.push_back(weight[0]);
    dfs(0,weight[0]);
    
    sort(ans.begin(),ans.end(),cmp);

    for(unsigned i=0;i<ans.size();i++){
        for(unsigned j=0;j<ans[i].size();j++){
            cout<<ans[i][j];
            if(j<ans[i].size()-1) cout<<" ";
        }
        cout<<endl;
    }
    return 0;
}
