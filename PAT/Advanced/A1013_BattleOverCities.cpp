#include<iostream>
#include<vector>
using namespace std;

vector< vector<int> > Grid(1001,vector<int>(1001,0));
vector<bool> visited(1001,false);
int N,M,K;
void DFS(int index){
    visited[index] = true;
    for(int i=1;i<=N;i++){
        if(Grid[index][i]==1 && visited[i]==false)
            DFS(i);
    }
}

int main()
{
    cin>>N>>M>>K;
    int a,b;
    for(int i=0;i<M;i++){
        cin>>a>>b;
        Grid[a][b] = 1;
        Grid[b][a] = 1;
    }
    int c;
    for(int i=0;i<K;i++){
        cin>>c;
        for(int j=1;j<=N;j++)
            visited[j]=false;
        visited[c] = true;
        int cnt = 0;
        for(int j=1;j<=N;j++){
            if(visited[j]==false){
                cnt++;
                DFS(j);
            }
        }
        cout<<cnt-1<<endl;
    }
    return 0;
}
