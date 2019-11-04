/*
问题描述
　　栋栋最近开了一家餐饮连锁店，提供外卖服务。
    随着连锁店越来越多，怎么合理的给客户送餐成
	为了一个急需解决的问题。栋栋的连锁店所在的
	区域可以看成是一个n×n的方格图(MealDeliver.png),
	方格的格点上的位置上可能包含栋栋的分店（绿
	色标注）或者客户（蓝色标注），有一些格点是
	不能经过的（红色标注）。方格图中的线表示可
	以行走的道路，相邻两个格点的距离为1。栋栋要
	送餐必须走可以行走的道路，而且不能经过红色标注的点。

　　送餐的主要成本体现在路上所花的时间，每一份
    餐每走一个单位的距离需要花费1块钱。每个客户
	的需求都可以由栋栋的任意分店配送，每个分店
	没有配送总量的限制。
　　
    现在你得到了栋栋的客户的需求，请问在最优的
	送餐方式下，送这些餐需要花费多大的成本。
输入格式
　　输入的第一行包含四个整数n, m, k, d，分别表示:
		m:方格图的大小
		n:栋栋的分店数量
		k:客户的数量
		d:不能经过的点的数量
　　接下来m行，每行两个整数xi, yi，表示栋栋的一个分店在方格图中的横坐标和纵坐标。
　　接下来k行，每行三个整数xi, yi, ci，分别表示每个客户在方格图中的横坐标、纵坐标和订餐的量。（注意，可能有多个客户在方格图中的同一个位置）
　　接下来d行，每行两个整数，分别表示每个不能经过的点的横坐标和纵坐标。
输出格式
　　输出一个整数，表示最优送餐方式下所需要花费的成本。
样例输入
    10 2 3 3
    1 1
    8 8
    1 5 1
    2 3 3
    6 7 2
    1 2
    2 2
    6 8
样例输出
    29
评测用例规模与约定
　　前30%的评测用例满足：1<=n <=20。
　　前60%的评测用例满足：1<=n<=100。
　　所有评测用例都满足：1<=n<=1000，1<=m, k, d<=n^2。可能有多个客户在同一个格点上。每个客户的订餐量不超过1000，每个客户所需要的餐都能被送到。
*/
#include<iostream>
#include<vector>
#include<queue>
using namespace std;
struct Pos{
    int x,y,val;
    Pos(int _x,int _y):x(_x),y(_y),val(0){}
    Pos():x(0),y(0),val(0){}
};
Pos dir[4] = {Pos(0,1),Pos(0,-1),Pos(1,0),Pos(-1,0)};

class Solution{
    vector< vector<int> > dist;
    vector< vector<bool> > visited;
    int n,m,k,d;
public:
    bool check(int x,int y){
        return (x>=1 && y >=1 && x <= n && y <= n);
    }
    void MealDeliver(){
        cin>>n>>m>>k>>d;
		//initial
        dist.resize(n+1);
        visited.resize(n+1);
        for(int i=0;i<n+1;i++){
            dist.at(i).resize(n+1);
            visited.at(i).resize(n+1);
            for(int j=0;j<n+1;j++){
                dist.at(i).at(j) = 0;
                visited.at(i).at(j) = false;
            }
        }//initial finished
        queue<Pos> stores;
        vector<Pos> holes,clients;
		//load data from cin
        Pos p;
        for(int i=0;i<m;i++){
            cin>>p.x>>p.y;
            stores.push(p);
            dist[p.x][p.y] = 0;
            visited[p.x][p.y] = true;
        }
        for(int i=0;i<k;i++){
            cin>>p.x>>p.y>>p.val;
            clients.push_back(p);
        }
        for(int i=0;i<d;i++){
            cin>>p.x>>p.y;
            dist[p.x][p.y] = -1;
            visited[p.x][p.y] = true;
        }
		//data load finished
		
        while(!stores.empty()){  //BFS
            p = stores.front();
            stores.pop();
            for(int i=0;i<4;i++){
                Pos nextp(p.x+dir[i].x,p.y+dir[i].y);
                if( check(nextp.x,nextp.y) && !visited[nextp.x][nextp.y] ){
                    visited[nextp.x][nextp.y] = true;
                    dist[nextp.x][nextp.y] = dist[p.x][p.y]+1;
                    stores.push(nextp);
                }
            } //for
        } //while
        long long cost = 0;
        for(int i=0;i<k;i++){
            cost += dist[clients[i].x][clients[i].y] * clients[i].val;
        }
        cout<<cost<<endl;
    }
};

int main()
{
    Solution s;
    s.MealDeliver();
    return 0;
}
