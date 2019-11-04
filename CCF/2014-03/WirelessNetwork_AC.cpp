#include<iostream>
#include<vector>
#include<queue>
#include<cstdio>
#include<cmath>
using namespace std;

typedef struct Point{
	double x,y;
	Point():x(0),y(0){}
	Point(int  _x,int _y):x(_x),y(_y){}
	double disquare(Point p){
		return (double)(sqrt(((p.x-x) * (p.x-x)) + ((p.y-y) * (p.y-y))));
	}
}Pos;
typedef pair<int,int> P;
class Solution{
	vector<int> G[202];
	vector<Pos> point;
	bool visited[202];
	long long N,M,K,R;
	void GetData(){
		cin>>N>>M>>K>>R;
		int n = N+M;
		Pos p;
		for(int i=0;i<n;i++){
			cin>>p.x>>p.y;
			point.push_back(Pos(p.x,p.y));
		}
		for(int i=0;i<n;i++){
			for(int j=0;j<i;j++){
				if(point[i].disquare(point[j]) <= (double)R){
					G[i].push_back(j);
					G[j].push_back(i);
				}
			}
		}
		for(int i=0;i<202;i++){
			visited[i] = false;
		}
	}
	
	int BFS(){
		queue<P> q;
		visited[0] = true;
		q.push(P(0,0));//开始
		while(!q.empty()){
			int now = q.front().first,step = q.front().second;q.pop();
			if(now==1) 
				return step;
			int size = G[now].size();
			for(int i=0;i<size;i++){
				int next = G[now][i];
				if(visited[next]) continue;
				visited[next] = true;
				q.push(P(next,step+1));
			}
		}
		
	}
public:
	void solve(){
		GetData();
		cout<<BFS()-1<<endl;
	}
};

int main()
{
	freopen("test.txt","r",stdin);
	Solution s;
	s.solve();
	return 0;
}