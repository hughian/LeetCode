#include "LeetCode.h"
class Solution {
public:
    int uniquePathsWithObstacles(vector< vector<int> >& obstacleGrid) {
        int m = obstacleGrid.size();
		int n = m > 0 ? obstacleGrid[0].size():0;
		vector< vector<int> > res(m+1,vector<int>(n+1,0));
		res[0][1] = 1;
		for(int i=1;i<m+1;i++){
			for(int j=1;j<n+1;j++){
				if(!obstacleGrid[i-1][j-1])
					res[i][j] = res[i-1][j] + res[i][j-1];
			}
		}
		return res[m][n];
    }
};

int main()
{
	vector<int> a(2);
	vector<vector<int> > v;
	v.push_back(a);
	Solution s;
	cout<<s.uniquePathsWithObstacles(v);
	return 0;
}
