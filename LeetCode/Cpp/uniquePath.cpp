#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
        vector< vector<int> > res(m);
        for(int i=0;i<m;i++){
            res[i].resize(n);
            for(int j=0;j<n;j++){
                if(i==0 || j==0)
                    res[i][j] = 1;
            }
        }
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                res[i][j] = res[i-1][j] + res[i][j-1];
        return res[m-1][n-1];
    }
};

int main()
{
    Solution s;
    s.uniquePaths(3,7);
    return 0;
}
