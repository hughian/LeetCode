/*��������
��������һ��R��C�еĵ�ͼ����ͼ��ÿһ�����������'#', '+', '-', '|', '.', 'S', 'T'�߸��ַ��е�һ�����ֱ��ʾ������˼��
����'#': �κ�ʱ����Ҷ������ƶ����˷���
����'+': ����ҵ�����һ�������һ���������������ҡ��ĸ��������ڵ�����һ����'#'�����ƶ�һ��
����'-': ����ҵ�����һ�������һ�����������ҡ������������ڵ�һ����'#'�����ƶ�һ��
����'|': ����ҵ�����һ�������һ�����������¡������������ڵ�һ����'#'�����ƶ�һ��
����'.': ����ҵ�����һ�������һ��ֻ�����¡��ƶ�һ������������ڵķ���Ϊ'#'������Ҳ������ƶ���
����'S': ��ҵĳ�ʼλ�ã���ͼ��ֻ����һ����ʼλ�á���ҵ�����һ�������һ���������������ҡ��ĸ��������ڵ�����һ����'#'�����ƶ�һ��
����'T': ��ҵ�Ŀ��λ�ã���ͼ��ֻ����һ��Ŀ��λ�á���ҵ�����һ����󣬿���ѡ���������Ҳ����ѡ�������������ƶ���
         ��������ƶ���һ���������������ҡ��ĸ��������ڵ�����һ����'#'�����ƶ�һ��
�������⣬��Ҳ����ƶ�����ͼ��
�������ҳ����������������ʵķ��������
����1. ��ҿ��Դӳ�ʼλ���ƶ����˷���
����2. ��Ҳ����ԴӴ˷����ƶ���Ŀ��λ�á�
�����ʽ
��������ĵ�һ�а�����������R ��C���ֱ��ʾ��ͼ���к�������(1 �� R, C �� 50)��
������������R��ÿ�ж�����C���ַ������Ǳ�ʾ��ͼ�ĸ��ӡ���ͼ��ǡ����һ��'S'��һ��'T'��
�����ʽ
�����������ڳ�ʼλ�þ��Ѿ����ܵ����յ��ˣ��������I'm stuck!��������˫���ţ�������Ļ�������������ʵķ���ĸ�����
��������
	5 5
	--+-+
	..|#.
	..|##
	S-+-T
	####.
�������
	2
����˵��
����������������ʵķ����ڵ�ͼ����'X'��ǳ����Ļ�����ͼ������ʾ��
����--+-+
����..|#X
����..|##
����S-+-T
����####X
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
            if(checkS(np) && !visitedS[np.x][np.y] && map[np.x][np.y]) //�ھ���Χ�ڣ�û�б����ʹ����Ҹõ�ɵ������'#'��;
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
            if(checkT(np,(i+2)%4) && !visitedT[np.x][np.y]) //�ھ���Χ�ڣ�û�б����ʹ����Ҹõ�ɵ�����Ҹ��ݹ���õ�����ƶ���ǰһ����
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
