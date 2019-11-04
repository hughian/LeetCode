#include<bits/stdc++.h>
#define maxn 1001
using namespace std;
int N,M,K,D;
char table[maxn][maxn];//store-'s' customer-'c' block-'x'
int dir[4][2]={{0,-1},{-1,0},{0,1},{1,0}};
bool vis[maxn][maxn];
map<int,int> cust;
long long ans;
struct Node{
	int r,c,s;
	Node(int r,int c,int s):r(r),c(c),s(s){}
	bool operator <(const Node &b)const{return s>b.s;}
};
priority_queue<Node> p;

inline bool checkFail(int r,int c){
	return r<1||r>N||c<1||c>N||table[r][c]=='x'||vis[r][c];
}

void bfs()
{
	int cnt=0;
	while(!p.empty()){
		if(cnt==K) return; 
		int r=p.top().r,c=p.top().c,s=p.top().s;p.pop();
		
		if(table[r][c]=='c'){
			cnt++;
			//printf("%d %d\n",cust[(r-1)*N+c]);
			ans+=(long long)s*cust[(r-1)*N+c];	
		}
		
		for(int i=0;i<4;i++){
			int nr=r+dir[i][0],nc=c+dir[i][1],ns=s+1;
			if(checkFail(nr,nc)) continue;
			//printf("n %d %d\n",nr,nc);
			vis[nr][nc]=true;
			p.push(Node(nr,nc,ns));
		}			
	}
}

void SOLVE()
{	
	ans=0LL;
	bfs();
	printf("%lld\n",ans);
}

void INPUT()
{
	int r,c,k;
	scanf("%d%d%d%d",&N,&M,&K,&D);
	for(int i=0;i<M;i++){
		scanf("%d%d",&r,&c);
		table[r][c]='s';
		vis[r][c]=true;
		p.push(Node(r,c,0));
	}
	for(int i=0;i<K;i++){
		scanf("%d%d%d",&r,&c,&k);
		table[r][c]='c';
		cust[(r-1)*N+c]=k;
	}
	for(int i=0;i<D;i++){
		scanf("%d%d",&r,&c);
		table[r][c]='x';
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
