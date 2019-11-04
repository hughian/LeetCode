#include<iostream>
#include<vector>
#include<stack>
using namespace std;
const int Inf = 1e9;
vector< vector<int> > cost(510,vector<int>(510,Inf));
vector< vector<int> > dist(510,vector<int>(510,Inf));
int N,M,S,D;
vector<int>  c(510,Inf);
vector<int>  d(510,Inf);
vector<int>  path(510,0);
void dijkstra(int st){
    vector<bool> s(510,false);
    for(int i=0;i<N;i++){
        path[i] = st;
        d[i] = dist[st][i];
        c[i] = cost[st][i];
    }
    c[st] = 0;
    d[st] = 0;
    s[st] = true;
    for(int i=1;i<N;i++){
        int minD = Inf;
        int minC = Inf;
        int u = st;
        for(int j=0;j<N;j++){
            if(!s[j] && (d[j] < minD || (d[j]==minD && c[j]<minC))){
                u = j;
                minD = d[j];
                minC = c[j];
            }
        }
        s[u] = true;
        for(int j=0;j<N;j++){
            if(!s[j] && dist[u][j] < Inf){
                int tmp = d[u] + dist[u][j];
                if(tmp<d[j]){
                    d[j] = tmp;
                    c[j] = c[u]+cost[u][j];
                    path[j] = u;
                }else if(tmp == d[j] && c[u]+cost[u][j]<c[j]){
                    d[j] = tmp;
                    c[j] = c[u]+cost[u][j];
                    path[j]=u;
                }
            }
        }
    }
    /*
    for(int i=0;i<N;i++)
        cout<<d[i]<<" ";
    cout<<endl;
    for(int i=0;i<N;i++)
        cout<<path[i]<<" ";
    cout<<endl;
    */
}
int main()
{
    int a,b,di,co;
    cin>>N>>M>>S>>D;
    for(int i=0;i<M;i++){
        cin>>a>>b>>di>>co;
        dist[a][b] = dist[b][a] = di;
        cost[a][b] = cost[b][a] = co;
    }
    dijkstra(S);
    int t=D;
    stack<int> stk;
    while(t != S){
        stk.push(t);
        t = path[t];
    }
    stk.push(S);
    while(!stk.empty()){
        cout<<stk.top()<<" ";
        stk.pop();
    }
    cout<<d[D]<<" "<<c[D];
    return 0;
}
