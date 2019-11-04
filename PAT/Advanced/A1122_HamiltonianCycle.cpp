#include<iostream>
#include<vector>

using namespace std;
vector< vector<int> > edge(210,vector<int>(210,0));
vector<int> used(210,0);
int n,m,k;
bool judge(vector<int>& path)
{
    int len = path.size();
    if(len != n+1) return false;
    if(path[0] != path[len-1]) return false;
    for(int i=0;i<=n;i++)
        used[i] = 0;
    for(int i=0;i<len-1;i++){
        used[path[i]]++;
        if(edge[path[i]][path[i+1]] != 1)
            return false;
    }
    for(int i=1;i<=n;i++){
        if(used[i] != 1)
            return false;
    }
    return true;
}

int main()
{
    cin>>n>>m;
    int a,b;
    for(int i=0;i<m;i++){
        cin>>a>>b;
        edge[a][b] = edge[b][a] = 1;
    }
    int tn;
    cin>>k;
    for(int i=1;i<=k;i++){
        cin>>tn;
        vector<int> path(tn,0);
        for(int j=0;j<tn;j++)
            cin>>path[j];
        
        if(judge(path))
            cout<<"YES"<<endl; //输出一定要看清楚题目
        else
            cout<<"NO"<<endl;
    }
    return 0;
}
