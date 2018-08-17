#include<iostream>
#include<vector>
using namespace std;
void print(vector<int> v){
	for(int i=0;i<v.size();i++)
		cout<<v[i]<<" ";
	cout<<endl;
}
void print2D(vector< vector<int> >& v)
{
	for(int i=0;i<v.size();i++)
		print(v[i]);
}
class Solution {
public:
    //
    void dfs(vector< vector<int> >& grid,int i,int j,int& sum){
        sum += 1;
        grid[i][j] = -1;
        int ii,jj;
		pair<int,int> dir[4] = {make_pair(-1,0),make_pair(1,0),make_pair(0,1),make_pair(0,-1)};
        for(int k=0;k<4;k++){
            ii = i + dir[k].first;
            jj = j + dir[k].second;
            if(ii>=0 && ii<grid.size() && jj>=0 && jj<=grid[0].size() && grid[ii][jj] == 1)
                dfs(grid,ii,jj,sum);
        }
    }
    int maxAreaOfIsland(vector<vector<int> >& grid) {
        int max = 0;
        int tmp=0;
		
        for(int i=0;i<grid.size();i++){
            for(int j=0;j<grid[0].size();j++){
                if(grid[i][j] == 1){
                    tmp = 0;
                    dfs(grid,i,j,tmp);
                }
                if(tmp>max) max = tmp;
            }
        }
        return max;
    }
};

int main()
{
	vector< vector<int> > grid(1); 
	int array[1][1]={{0}};
	for(int i=0;i<1;i++){
		grid[i].resize(1);
		for(int j=0;j<1;j++)
			grid[i][j] = array[i][j];
	}
	print2D(grid);
	Solution a;
	cout<<a.maxAreaOfIsland(grid)<<endl<<endl;
	print2D(grid);
	return 0;
}