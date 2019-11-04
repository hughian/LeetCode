#include<bits/stdc++.h>
#define inf 0x3f3f3f3f
using namespace std;
const int maxn=1e5+5;
int n,m;
int ma[maxn];
bool in[maxn]={0};
struct Edge
{
    int u,v,w;
    Edge(int uu,int vv,int ww){
        u=uu,v=vv,w=ww;
    }
};
queue<int> que;
vector<Edge> edge;
vector<int> ve[maxn];
void bfs(int s)
{
    que.push(s);
    in[s]=true;
    ma[s]=0;
    while(!que.empty()){
        int u=que.front();que.pop(); //出队列
        in[u]=false;
		
        for(int i=0;i<ve[u].size();i++){
            int e=ve[u][i];//e为邻接边编号 
            int v=edge[e].v;
            int temp=max(ma[u],edge[e].w);
            if(temp<ma[v]){//动态规划的思想 
                ma[v]=temp;
                if(!in[v]){
                    que.push(v);
                    in[v]=true;
                }
            }
        }
    }
}
int main()
{
    cin>>n>>m;
    fill(ma+1,ma+n+1,inf);
    int u,v,w;
    while(m--){
        scanf("%d%d%d",&u,&v,&w);
        edge.push_back(Edge(u,v,w));
        edge.push_back(Edge(v,u,w));
        ve[u].push_back(edge.size()-2);
        ve[v].push_back(edge.size()-1);
    }
    /*for(int i=1;i<=n;i++){
        for(int j=0;j<ve[i].size();j++)
            cout<<ve[i][j]<<' ';
        cout<<endl;
    }*/
    bfs(1);
    cout<<ma[n]<<endl;
    return 0;
}