#include<iostream>
using namespace std;

class Solution{
public:
    void Painting(){
        int canvas[101][101];
        int i,j;
        int cnt = 0;
        for(i=0;i<101;i++)
            for(j=0;j<101;j++)
                canvas[i][j] = 0;
        int n,k;
        cin>>n;
        int x0,y0,x1,y1;
        for(k=0;k<n;k++){
            cin>>x0>>y0>>x1>>y1;
            for(i=x0+1;i<=x1;i++)
                for(j=y0+1;j<=y1;j++)
                    canvas[i][j] = 1;
        }
        for(i=0;i<101;i++)
            for(j=0;j<101;j++)
                if(canvas[i][j] == 1)
                    cnt++;
        cout<<cnt<<endl;
    }
};
int main()
{
    Solution s;
    s.Painting();
    return 0;
}
