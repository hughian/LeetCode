/*问题描述
　　给定一个R行C列的地图，地图的每一个方格可能是'#', '+', '-', '|', '.', 'S', 'T'七个字符中的一个，分别表示如下意思：
　　'#': 任何时候玩家都不能移动到此方格；
　　'+': 当玩家到达这一方格后，下一步可以向【上下左右】四个方向相邻的任意一个非'#'方格移动一格；
　　'-': 当玩家到达这一方格后，下一步可以向【左右】两个方向相邻的一个非'#'方格移动一格；
　　'|': 当玩家到达这一方格后，下一步可以向【上下】两个方向相邻的一个非'#'方格移动一格；
　　'.': 当玩家到达这一方格后，下一步只能向【下】移动一格。如果下面相邻的方格为'#'，则玩家不能再移动；
　　'S': 玩家的初始位置，地图中只会有一个初始位置。玩家到达这一方格后，下一步可以向【上下左右】四个方向相邻的任意一个非'#'方格移动一格；
　　'T': 玩家的目标位置，地图中只会有一个目标位置。玩家到达这一方格后，可以选择完成任务，也可以选择不完成任务继续移动。
         如果继续移动下一步可以向【上下左右】四个方向相邻的任意一个非'#'方格移动一格。
　　此外，玩家不能移动出地图。
　　请找出满足下面两个性质的方格个数：
　　1. 玩家可以从初始位置移动到此方格；
　　2. 玩家不可以从此方格移动到目标位置。
输入格式
　　输入的第一行包括两个整数R 和C，分别表示地图的行和列数。(1 ≤ R, C ≤ 50)。
　　接下来的R行每行都包含C个字符。它们表示地图的格子。地图上恰好有一个'S'和一个'T'。
输出格式
　　如果玩家在初始位置就已经不能到达终点了，就输出“I'm stuck!”（不含双引号）。否则的话，输出满足性质的方格的个数。
样例输入
	5 5
	--+-+
	..|#.
	..|##
	S-+-T
	####.
样例输出
	2
样例说明
　　如果把满足性质的方格在地图上用'X'标记出来的话，地图如下所示：
　　--+-+
　　..|#X
　　..|##
　　S-+-T
　　####X
*/

#include<iostream>
#include<vector>
#include<map>
using namespace std;

struct Pos{
    int x,y;
    Pos(int _x,int _y):x(_x),y(_y){};
    Pos():x(0),y(0){}
};
/*
 *      (-1,0)
 *  (0,-1)  (0,1)
 *      (1,0)
 */
//   up left down right
//bit 0  0    0   0
//'#' 0  0    0   0  = 0
//'+' 1  1    1   1  = 15
//'S' 1  1    1   1  = 15
//'T' 1  1    1   1  = 15
//'-' 0  1    0   1  = 5
//'|' 1  0    1   0  = 10
//'.' 0  0    1   0  = 2   
Pos dir[4] = {Pos(0,1),Pos(1,0),Pos(0,-1),Pos(-1,0)};
class Solution{
    vector< vector<char> > map;
    vector< vector<bool> > visitedT; //true visited
    vector< vector<bool> > visitedS;
    ::map<char,int> Trans;
    int r_,c_;
    int sx,sy;//S pos;
    int tx,ty;//T pos;
public:
    Solution(int r,int c):r_(r),c_(c),sx(0),sy(0),tx(0),ty(0){
        map.resize(r);
        visitedT.resize(r);
        visitedS.resize(r);
        for(int i=0;i<r;i++){
            map.at(i).resize(c);
            visitedT.at(i).resize(c);
            visitedS.at(i).resize(c);
        }
        Trans['S'] = Trans['T'] = Trans['+'] = 15;
        Trans['-'] = 5;
        Trans['|'] = 10;
        Trans['.'] = 2;
        Trans['#'] = 0;
    }
public:
    void Imstuck(){
        getMapData();
        Pos ps(sx,sy);
        Pos pt(tx,ty);
        DFS_S(ps);
        DFS_T(pt);
        if(!visitedT[sx][sy])
            cout<<"I'm stuck!"<<endl;
        else{
            int ans = 0;
            for(int i=0;i<r_;i++)
                for(int j=0;j<c_;j++)
                    if(visitedS[i][j] && !visitedT[i][j])
                        ans ++;
            cout<<ans<<endl;
        }
    }
private:
    void DFS_S(Pos p){
        visitedS[p.x][p.y]=true;
        for(int i=0;i<4;i++){
            if(!(map[p.x][p.y]>>i&1)) continue;
            Pos np(p.x+dir[i].x,p.y+dir[i].y);
            if(checkS(np) && !visitedS[np.x][np.y] && map[np.x][np.y]) //在矩阵范围内，没有被访问过，且该点可到达（不是'#'）;
                DFS_S(np);
        }
    }
    inline bool checkS(Pos p){
        return (p.x>=0 && p.x<r_ && p.y >= 0 && p.y < c_);
    }
    inline bool checkT(Pos p,int d){
        return (p.x>=0 && p.x<r_ && p.y >= 0 && p.y <c_ && map[p.x][p.y] && (map[p.x][p.y]>>d&1));
    }
    void DFS_T(Pos p){
        visitedT[p.x][p.y] = true;
        for(int i=0;i<4;i++){
            Pos np(p.x+dir[i].x,p.y+dir[i].y);
            if(checkT(np,(i+2)%4) && !visitedT[np.x][np.y]) //在矩阵范围内，没有被访问过，且该点可到达，并且根据规则该点可以移动到前一个点
                DFS_T(np);
        }
    }
    void getMapData(){
        char c;
        for(int i=0;i<r_;i++)
            for(int j=0;j<c_;j++){
                visitedT[i][j] = false;
                visitedS[i][j] = false;
                cin>>c;
                map[i][j] = Trans[c];
                if(c == 'T'){
                    tx = i;ty = j;
                    visitedT[i][j] = true;
                }
                if(c == 'S'){
                    sx = i;sy = j;
                    visitedS[i][j] = true;
                }
            }
    }
};

int main(void)
{
    int r,c;
    cin>>r>>c;
    Solution s(r,c);
    s.Imstuck();
    return 0;
}
