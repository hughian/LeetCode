/*问题描述
　　消除类游戏是深受大众欢迎的一种游戏，游戏在一个包含有n行m列的游戏棋盘上进行。
    棋盘的每一行每一列的方格上放着一个有颜色的棋子，
	当一行或一列上有连续三个或更多的相同颜色的棋子时，这些棋子都被消除。
	当有多处可以被消除时，这些地方的棋子将同时被消除。
　　现在给你一个n行m列的棋盘，棋盘中的每一个方格上有一个棋子，请给出经过一次消除后的棋盘。
　　请注意：一个棋子可能在某一行和某一列同时被消除。
输入格式
　　输入的第一行包含两个整数n, m，用空格分隔，分别表示棋盘的行数和列数。
　　接下来n行，每行m个整数，用空格分隔，分别表示每一个方格中的棋子的颜色。颜色使用1至9编号。
输出格式
　　输出n行，每行m个整数，相邻的整数之间使用一个空格分隔，表示经过一次消除后的棋盘。
    如果一个方格中的棋子被消除，则对应的方格输出0，否则输出棋子的颜色编号。
*/

#include<iostream>
using namespace std;
int main(void)
{
	int n,m;
	int i,j;
	cin>>n;
	cin>>m;
	
	int mat[n][m];
	int flg[n][m];
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			cin>>mat[i][j];
			flg[i][j] = 0;
		}
	}
	//check row
	for(i=0;i<n;i++){
		for(j=1;j<m-1;j++){
			if(mat[i][j] == mat[i][j-1] && mat[i][j] == mat[i][j+1]){
				flg[i][j] = 1;flg[i][j-1] = 1;flg[i][j+1] = 1;
			}
		}
	}
	//check column
	for(j=0;j<m;j++){
		for(i=1;i<n-1;i++){
			if(mat[i][j] == mat[i-1][j] && mat[i][j] == mat[i+1][j]){
				flg[i][j]=1;flg[i-1][j] = 1;flg[i+1][j]=1;
			}
		}
	}
	//disolve
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			if(flg[i][j] == 1)
				mat[i][j] = 0;
		}
	}
	//print result
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			cout<<mat[i][j]<<" ";
		}
		cout<<endl;
	}
		
	
}