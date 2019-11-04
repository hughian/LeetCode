#include<bits/stdc++.h>
#define maxv 202
using namespace std;
struct Point{
	double x,y;
	
	inline void read(){scanf("%lf%lf",&x,&y);}
	double Length(Point b){return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y));}
}p[maxv];

double R;
vector<int> G[maxv];
int N,M,K,star=0,fin=1,vis[maxv];
typedef pair<int,int> P;//<node,step>

inline bool checkFail(int next){
	return vis[next];
}

int bfs()
{
	queue<P> q;
	vis[star]=true;
	q.push(P(star,0));
	
	while(!q.empty()){
		int now=q.front().first,step=q.front().second;q.pop();
		if(now==fin) return step;
		int size=G[now].size();
		for(int i=0;i<size;i++){
			int next=G[now][i];
			if(checkFail(next)) continue;
			vis[next]=true;
			q.push(P(next,step+1));
		}
	}
}

inline void SOLVE(){
	printf("%d\n",bfs()-1);
}

void INPUT()
{
	scanf("%d%d%d%lf",&N,&M,&K,&R);
	int n=N+M;
	for(int i=0;i<n;i++) p[i].read();
	for(int i=0;i<n;i++)
		for(int j=0;j<i;j++)
			if(p[i].Length(p[j])<=R){
				G[i].push_back(j);
				G[j].push_back(i);
			}		
}

void MAIN()
{
	INPUT();
	SOLVE();
}

int main()
{
	freopen("test.txt","r",stdin);
	MAIN();
	return 0;
}
