#include<iostream>
#include<vector>
#include<queue>
using namespace std;
struct Pos{
    int lv,x,y;
    Pos(int _lv,int _x,int _y):lv(_lv),x(_x),y(_y){}
	Pos& operator = (const Pos p){
		this->lv = p.lv;
		this->x = p.x;
		this->y = p.y;
		return *this;
	}
};
Pos dir[6] = {Pos(-1,0,0),Pos(1,0,0),Pos(0,-1,0),Pos(0,1,0),Pos(0,0,-1),Pos(0,0,1)};
int shape[61][1287][129] = {{{0}}};
bool used[61][1287][129] = {{{false}}};
long long cnt = 0;
long long maxC= 0;
int M,N,L,T;
//dfs 递归深度太大，会段溢出
void dfs(int lv,int x,int y){
    used[lv][x][y] = true;
    cnt++;
    int nlv,nx,ny;
    for(int i=0;i<6;i++){
        nlv = lv+dir[i].lv;
        nx = x+dir[i].x;
        ny = y+dir[i].y;
        if(nlv<0 || nx<0 || ny<0 || nlv>=L || nx >=M ||ny >= N)
            continue;
        if(shape[nlv][nx][ny]==1 && !used[nlv][nx][ny])
            dfs(nlv,nx,ny);
    }
}

void bfs(Pos p){
	queue<Pos> q;
	q.push(p);
	cnt = 0;
	used[p.lv][p.x][p.y] = true;
	while(!q.empty()){
		Pos tp = q.front();q.pop();
		cnt ++;
		//cout<<cnt<<":("<<tp.lv<<","<<tp.x<<","<<tp.y<<")"<<endl;
		Pos np(0,0,0);
		for(int i=0;i<6;i++){
			np.lv = tp.lv + dir[i].lv;
			np.x = tp.x + dir[i].x;
			np.y = tp.y + dir[i].y;
			if(np.lv<0 || np.x<0 || np.y<0 || np.lv>=L || np.x >=M ||np.y >= N)
				continue;
			if(shape[np.lv][np.x][np.y]==1 && used[np.lv][np.x][np.y]==false){
				q.push(np);
				used[np.lv][np.x][np.y] = true;
			}
		}
	}
}
int main()
{
    cin>>M>>N>>L>>T;
    for(int k=0;k<L;k++){
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                cin>>shape[k][i][j];
				used[k][i][j] = false;
			}
        }
    }
    for(int k=0;k<L;k++){
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                if(shape[k][i][j]==1 && !used[k][i][j]){
                    cnt = 0;
                    bfs(Pos(k,i,j));
                    if(cnt>=T)
                        maxC += cnt;
                }
            }
        }
    }
    cout<<maxC;
    return 0;
}
