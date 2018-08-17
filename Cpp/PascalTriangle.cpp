#include<iostream>
#include<vector>
using namespace std;
	vector< vector<int> > generate(int numRows) {
        vector< vector<int> > ans(numRows);
        for(int i=0;i<numRows;i++){
            ans[i].resize(i+1);
            ans[i][0]=ans[i][i] = 1;
            if(i>1){
                for(int j=1;j<i;j++){
                    ans[i][j] = ans[i-1][j-1] + ans[i-1][j];
                }
            }
        }
		for(int i=0;i<numRows;i++){
			for(int j=0;j<i+1;j++)
				cout<<ans[i][j]<<" ";
			cout<<endl;
		}
        return ans;
    }
	
int main()
{
	generate(5);
	return 0;
}