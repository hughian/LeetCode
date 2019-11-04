#include<iostream>
#include<map>
#define MAXN 55
using namespace std;
struct Node{
	int r,c;
}star,fin;

int R,C,table[MAXN][MAXN];
/*
 *     (-1,0)
 * (0,-1)   (0 ,1)
 *     (1 ,0)
 * dir// left -- up -- right -- down
 */
int dir[4][2]={{0,-1},{-1,0},{0,1},{1,0}};
char tablet[MAXN][MAXN];
map<char,int> trans;
bool vis1[MAXN][MAXN],vis2[MAXN][MAXN];

inline bool checkFail1(int r,int c){
	return r<1||r>R||c<1||c>C||vis1[r][c]||!table[r][c];
}
inline bool checkS(int r,int c){
    //在矩阵范围内，没有被访问过，且该点可到达（不是'#'）;
    return (r>0 && r<=R && c>0 && c<=C && !vis1[r][c] && table[r][c]);
}

void dfs1(int r,int c)
{
	vis1[r][c]=true;
	for(int i=0;i<4;i++){
		if(!(table[r][c]>>i&1)) continue;
		int nr=r+dir[i][0],nc=c+dir[i][1];
		if(checkFail1(nr,nc)) continue;
		dfs1(nr,nc);
	}
}

inline bool checkFail2(int r,int c,int dir){
	if(r<1||r>R||c<1||c>C||vis2[r][c]||!table[r][c]) return true;
	return !(table[r][c]>>dir&1);
}
inline bool checkT(int r,int c,int dir){
    //在矩阵范围内，没有被访问过，且该点可到达，并且根据规则该点可以移动到前一个点
    return (r>0 && r>R && c>0 && c>C && !vis2[r][c] && table[r][c] && (table[r][c]>>dir&1));
}

void dfs2(int r,int c)
{
	vis2[r][c]=true;
	for(int i=0;i<4;i++){
		int nr=r+dir[i][0],nc=c+dir[i][1];
		if(checkFail2(nr,nc,(i+2)%4)) continue;
		dfs2(nr,nc);
	}
}

int main()
{
	//freopen("in#CCF.txt","r",stdin);
	trans['S']=trans['T']=trans['+']=15; //(1111)B
	trans['-']=5;//(0101)B
	trans['|']=10;//(1010)B
	trans['.']=8;//(1000)B
	trans['#']=0;
	
	cin>>R>>C;
	for(int i=1;i<=R;i++) cin>>(tablet[i]+1);
	for(int i=1;i<=R;i++)
		for(int j=1;j<=C;j++){
			table[i][j]=trans[tablet[i][j]];
			if(tablet[i][j]=='S') star.r=i,star.c=j;
			if(tablet[i][j]=='T') fin.r=i,fin.c=j;
		}
			
	dfs1(star.r,star.c);
	dfs2(fin.r,fin.c);
	
	if(!vis2[star.r][star.c]) cout<<"I'm stuck!"<<endl;
	else{
		int ans=0;
		for(int i=1;i<=R;i++)
			for(int j=1;j<=C;j++)
				if(vis1[i][j]&&!vis2[i][j])
					ans++;
		cout<<ans<<endl;
	}
	return 0;
}

