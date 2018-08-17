#include "LeetCode.h"

class Solution {
public:
	int min(int a,int b){
		return a<b?a:b;
	}
    int minPathSum(vector< vector<int> >& grid) {
        int m = grid.size();
        int n = m>0?grid[0].size():0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
				if(i==0 && j==0)
					continue;
                else if(i>0 && j==0)
                    grid[i][j] += grid[i-1][j];
                else if(i==0 && j>0)
                    grid[i][j] += grid[i][j-1];
                else
                    grid[i][j] += this->min(grid[i-1][j],grid[i][j-1]);
            }
        }
        return grid[m-1][n-1];
    }
};

int main()
{
	int a[][3] = {{1,3,1},{1,5,1},{4,2,1}};
	vector< vector<int> > v(3,vector<int>(3,0));
	for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
			v[i][j]=a[i][j];
		}
	}
	Solution s;
	cout<<s.minPathSum(v);
	return 0;
}