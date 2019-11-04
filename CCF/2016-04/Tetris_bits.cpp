/*问题描述
　　俄罗斯方块是俄罗斯人阿列克谢·帕基特诺夫发明的一款休闲游戏。
　　游戏在一个15行10列的方格图上进行，方格图上的每一个格子可能已经放置了方块，或者没有放置方块。每一轮，都会有一个新的由4个小方块组成的板块从方格图的上方落下，玩家可以操作板块左右移动放到合适的位置，当板块中某一个方块的下边缘与方格图上的方块上边缘重合或者达到下边界时，板块不再移动，如果此时方格图的某一行全放满了方块，则该行被消除并得分。
　　在这个问题中，你需要写一个程序来模拟板块下落，你不需要处理玩家的操作，也不需要处理消行和得分。
　　具体的，给定一个初始的方格图，以及一个板块的形状和它下落的初始位置，你要给出最终的方格图。
输入格式
　　输入的前15行包含初始的方格图，每行包含10个数字，相邻的数字用空格分隔。如果一个数字是0，表示对应的方格中没有方块，如果数字是1，则表示初始的时候有方块。输入保证前4行中的数字都是0。
　　输入的第16至第19行包含新加入的板块的形状，每行包含4个数字，组成了板块图案，同样0表示没方块，1表示有方块。输入保证板块的图案中正好包含4个方块，且4个方块是连在一起的（准确的说，4个方块是四连通的，即给定的板块是俄罗斯方块的标准板块）。
　　第20行包含一个1到7之间的整数，表示板块图案最左边开始的时候是在方格图的哪一列中。注意，这里的板块图案指的是16至19行所输入的板块图案，如果板块图案的最左边一列全是0，则它的左边和实际所表示的板块的左边是不一致的（见样例）
输出格式
　　输出15行，每行10个数字，相邻的数字之间用一个空格分隔，表示板块下落后的方格图。注意，你不需要处理最终的消行。
样例输入
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 1 0 0 0
    1 1 1 0 0 0 1 1 1 1
    0 0 0 0 1 0 0 0 0 0
    0 0 0 0
    0 1 1 1
    0 0 0 1
    0 0 0 0
    3
样例输出
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 1 0 0 0
    1 1 1 1 1 1 1 1 1 1
    0 0 0 0 1 1 0 0 0 0
*/
#include<iostream>
#include<vector>
using namespace std;

class Solution{
	vector< vector<int> > mat;
	vector< vector<int> > shape;
	vector<int> matBit;
	vector<int> shapeBit;
	int offset; 
public:
	Solution(){
		mat.resize(15);
		for(int i=0;i<15;i++)
			mat[i].resize(10);
		matBit.resize(4);
		shapeBit.resize(4);
		shape.resize(4);
		for(int i=0;i<4;i++){
			matBit.at(i) = 0;
			shapeBit.at(i) = 0;
			shape[i].resize(4);
		}

	}
    void print(vector< vector<int> > &a,int m,int n){
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++)
				cout<<a.at(i).at(j)<<" ";
			cout<<endl;
		}
	}
	void bitprint(int x){
		int i = 15;
		while(i >= 0){
			cout<<(int)((x>>i) & 0x1);
			i--;
		}
		cout<<endl;
	}
	void bit2mat(int left){
		int j = left, i;
		for(;j < left+4;j++){
			i = 14;
			while(i>=0){
				if((matBit[j-left] >> i) & 0x1)
					mat.at(i).at(j) = 1;
				i--;
			}
		}
	}
	bool CanMoveStep(){
		for(int j=0;j<4;j++){
		    if(((shapeBit[j] << 1) & matBit[j]))
				return false;
			if((shapeBit[j]<<1) > (15<<11)) //保证不会移动出方格区域
				return false;
		}
		for(int j=0;j<4;j++){
			shapeBit[j] = shapeBit[j] << 1;
		}
	    return true;
	}
    void Tetris(void)
    {
        int i,j;
		//get data from cin
        for(i=0;i<15;i++)
            for(j=0;j<10;j++)
                cin>>mat[i][j];
        for(i=0;i<4;i++)
            for(j=0;j<4;j++)
                cin>>shape[i][j];
        cin>>offset;
		//data loaded
		
		//matrix to bit
        int leftAlign,rightAlign;
        for(j=0;j<4;j++){
			for(i=0;i<4;i++){
				if(shape.at(i).at(j) == 1)
					shapeBit.at(j) |= (1<<i);
			}
			//bitprint(shapeBit[j]);
		}
		
        leftAlign = offset-1;
        rightAlign = leftAlign + 3;
        for(j=leftAlign;j<=rightAlign;j++){
            for(i=0;i<15;i++){
                if(mat.at(i).at(j)==1)
					matBit[j-leftAlign] |= (1<<i);
            }
			//cout<<matBit[j-leftAlign]<<endl;
			//bitprint(matBit[j-leftAlign]);
        }
		//finished mat2bit
		//arrange
		for(i=0;i<15;i++){
			if(CanMoveStep())
				continue;
			break;
		}
		//cout<<shapeBit[0]<<endl<<shapeBit[1]<<endl<<shapeBit[2]<<endl<<shapeBit[3]<<endl;
		//finished
		//bit2mat
		for(j=0;j<4;j++){
			matBit[j] = matBit[j] | shapeBit[j];
		}
		bit2mat(leftAlign);
		print(mat,15,10);
    }
};

int main()
{
    Solution s;
    s.Tetris();
    return 0;
}
