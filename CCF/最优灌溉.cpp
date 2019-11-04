#include<bits/stdc++.h>
#define maxe 100010
#define maxv 1010
using namespace std;
struct Edge{
	int u,v;
	int cost;
	
	inline void read(){scanf("%d%d%d",&u,&v,&cost);}
	inline void prints(){printf("%d %d %d\n",u,v,cost);}
	bool operator <(const Edge &b)const{return cost<b.cost;}
}es[maxe];

int n,m;

int fa[maxv];
inline void init(int v)
{for(int i=0;i<=v;i++) fa[i]=i;}
int find_fa(int x)
{return fa[x]==x?x:fa[x]=find_fa(fa[x]);}

int Kruskal(){
	sort(es,es+m);
	init(n);
	int res=0;
	for(int i=0;i<m;i++){
		Edge e=es[i];
		int fa_u=find_fa(e.u),fa_v=find_fa(e.v);
		if(fa_u==fa_v) continue;
		fa[fa_u]=fa_v;
		res+=e.cost;
	}
	return res;
}

inline void SOLVE(){
	printf("%d\n",Kruskal());
}

void INPUT()
{
	scanf("%d%d",&n,&m);
	for(int i=0;i<m;i++) es[i].read();
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
