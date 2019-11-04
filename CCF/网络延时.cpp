#include<bits/stdc++.h>
#define maxv 20010
using namespace std;
int N,M;
bool vis[maxv];
vector<int> G[maxv];
typedef pair<int,int> P;//<node,distance>

inline bool checkFail(int next){
	return vis[next];
}

P bfs(int root)
{
	P ret;ret.first=ret.second=0;

	queue<P> q;
	q.push(P(root,0));
	vis[root]=true;
	while(!q.empty()){
		int now=q.front().first,dist=q.front().second;q.pop();
		if(ret.second<dist) ret.first=now,ret.second=dist;
		int size=G[now].size();
		for(int i=0;i<size;i++){
			int next=G[now][i];
			if(checkFail(next)) continue;
			vis[next]=true;
			q.push(P(next,dist+1));
		}
	}
	
	return ret;
}


void SOLVE()
{
	memset(vis,0,sizeof vis);
	int root=bfs(1).first;
	memset(vis,0,sizeof vis);
	int ans=bfs(root).second;
	printf("%d\n",ans);
}

void INPUT()
{
	int u;
	scanf("%d%d",&N,&M);
	for(int i=2;i<=N;i++){
		scanf("%d",&u);
		G[u].push_back(i);
		G[i].push_back(u);
	}
	for(int i=1;i<=M;i++){
		scanf("%d",&u);
		G[u].push_back(i+N);
		G[i+N].push_back(u);		
	}
}

void MAIN()
{
	INPUT();
	SOLVE();
}

int main()
{
	//freopen("in#CCF.txt","r",stdin);
	//freopen("out#CCF.txt","w",stdout);
	MAIN();
	return 0;
}
