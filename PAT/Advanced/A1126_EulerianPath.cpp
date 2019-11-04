#include<iostream>
#include<vector>

using namespace std;
vector< vector<int> > edge(501);
vector<bool> visited(501,false);
void dfs(int idx){
    visited[idx] = true;
    for(unsigned i=0;i<edge[idx].size();i++){
        if(!visited[edge[idx][i]])
            dfs(edge[idx][i]);
    }
}
int main()
{
    int N,M;
    cin>>N>>M;
    int a,b;
    for(int i=0;i<M;i++){
        cin>>a>>b;
        edge[a].push_back(b);
        edge[b].push_back(a);
    }
    int degree;
    int even = 0,odd = 0;
    for(int i=1;i<=N;i++){
        degree = edge[i].size();
        if(degree%2==0)
            even++;
        else
            odd++;
        cout<<degree<<(i<N?" ":"\n");
    }
    dfs(1);
    bool flag = true;
    for(int i=1;i<=N;i++)
        if(visited[i]==false){
            flag = false;
            break;
        }
    if(even==N && flag) //有欧拉回路的图必须是连通图
        cout<<"Eulerian";
    else if(odd==2 && flag)
        cout<<"Semi-Eulerian";
    else
        cout<<"Non-Eulerian";
    return 0;
}
