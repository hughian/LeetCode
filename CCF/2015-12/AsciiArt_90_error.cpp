#include<iostream>
#include<vector>
using namespace std;

class Solution{
	vector< vector<char> > canvas;
	vector< vector<char> > flg;
	int n_, m_;
public:
	Solution(int n,int m):n_(n),m_(m){
		canvas.resize(n);
		flg.resize(n);
		for(int i=0;i<n;i++){
			canvas.at(i).resize(m);
			flg.at(i).resize(m);
			for(int j=0;j<m;j++)
				canvas.at(i).at(j) = '.';
		}
	}
public:
    void print(){
        for(int i=n_-1;i>=0;i--){
            for(int j=0;j<m_;j++)
                cout<<canvas.at(i).at(j);
            cout<<endl;
        }
    } 
    void drawLine(int x[],int y[]){
        int i,j;
        int begin,end;
        if(x[0] == x[1]){//Vertical
            j = x[0];
            begin = y[0]>y[1]?y[1]:y[0];
            end = y[0]+y[1] - begin;
            for(i=begin;i<=end;i++){
                if(canvas.at(i).at(j) == '-')
                    canvas.at(i).at(j) = '+';
                else
                    canvas.at(i).at(j) = '|';
            }
        }
        if(y[0] == y[1]){//Horizontal
            i = y[0];
            begin = x[0]>x[1]?x[1]:x[0];
            end = x[0]+x[1]-begin;
            for(j=begin;j<=end;j++){
                if(canvas.at(i).at(j) == '|')
                    canvas.at(i).at(j) ='+';
                else
                    canvas.at(i).at(j) = '-';
            }
        }
    }
	void resetFlg(){
		for(int i=0;i<n_;i++)
			for(int j=0;j<m_;j++)
				flg[i][j] = 0;
	}
    bool toStop(int i,int j){
		if(i<0 || j<0 || i>n_-1 || j>m_-1)
			return true;
        if(flg.at(i).at(j) == 1)
			return true;
		if( canvas[i][j] == '|' || canvas[i][j] =='-' || canvas[i][j]=='+')
            return true;
        return false;
    }
    void fillBlank(int x,int y,char ch){
        //if(x<0 || y<0 || x>=n || y>=m)
          //  return;
        if(!toStop(x,y)){
            canvas.at(x).at(y) = ch;
			flg[x][y] = 1;
			fillBlank(x+1,y,ch);
			fillBlank(x-1,y,ch);
			fillBlank(x,y+1,ch);
			fillBlank(x,y-1,ch);
		}
    }
    void AsciiArt(int q)
    {
        int i,j;
        int cmd,x[2],y[2];
        char c;
        for(i=0;i<q;i++){
            cin>>cmd;
            if(cmd == 0){
                cin>>x[0];cin>>y[0];
                cin>>x[1];cin>>y[1];
                drawLine(x,y);
            }
            if(cmd == 1){
                cin>>y[0];cin>>x[0];
                cin>>c;
				resetFlg();
                fillBlank(x[0],y[0],c);
            }
        }
        print();
 
     }
};

int main(void)
{
	int n,m,q;
	cin>>m>>n>>q;
    Solution a(n,m);
    a.AsciiArt(q);
    return 0;
}
