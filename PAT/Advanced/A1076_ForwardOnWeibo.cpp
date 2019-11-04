#include<iostream>
#include<vector>
#include<queue>
using namespace std;

vector< vector<int> > edge(1010,vector<int>(1010,0));
vector<bool> used(1010,false);
int N,L;
int bfs(int v){
    used[v] = true;
    queue<int> q;
    int lv = 0,cnt = -1;
    q.push(v);
    q.push(0);
    while(q.size()>1){
        int t = q.front();q.pop();
        if(t==0){
            lv++;
            if(lv>L) break;
            else q.push(0);
        }else{
            cnt++;
            //cout<<t<<" ";
            for(int j=1;j<=N;j++){
                if(edge[t][j]==1 && !used[j]){
                    used[j]= true;
                    q.push(j);
                }
            }
        }
    }
    //cout<<endl;
    return cnt;
}
int main()
{
    cin>>N>>L;
    int m,t,k;
    for(int i=1;i<=N;i++){
        cin>>m;
        for(int j=0;j<m;j++){
            cin>>t;
            edge[t][i] = 1;
        }
    }
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>t;
        for(int j=1;j<=N;j++)
            used[j] = false;
        cout<<bfs(t)<<endl;
    }
}
