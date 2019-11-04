#include<iostream>

using namespace std;
int img[1001][1001];
class Solution{
public:
    void ImageRotation(){
        int n,m;
        cin>>n>>m;
        int i,j;
        for(i=0;i<n;i++)
            for(j=0;j<m;j++)
                cin>>img[i][j];
			
        for(j=m-1;j>=0;j--){
            for(i=0;i<n;i++){
                cout<<img[i][j]<<" ";
            }
            cout<<endl;
        }
    }
};

int main()
{
    Solution s;
    s.ImageRotation();
    return 0;
}
