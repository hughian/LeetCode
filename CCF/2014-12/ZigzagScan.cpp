#include<iostream>
#include<vector>
using namespace std;
class Solution{
    vector< vector<int> > mat;
public:
    void ZGlyphScan(){
        int n;
        cin>>n;
        mat.resize(n);
        for(int i=0;i<n;i++)
            mat.at(i).resize(n);
        int i,j;
        for(i=0;i<n;i++)
            for(j=0;j<n;j++)
                cin>>mat[i][j];
        int num = 0;
        while(num < 2*n-1){
            if(num%2){
                for(i=0;i<=num;i++){
                    if(i>=n || num-i>=n)
                        continue;
                    cout<<mat[i][num-i]<<" ";
                }
            }else{
                for(j=0;j<=num;j++){
                    if(j>=n || num-j>=n)
                        continue;
                    cout<<mat[num-j][j]<<" ";
                }
            }
            num++;
        }
    }
};
int main()
{
    Solution s;
    s.ZGlyphScan();
    return 0;
}
