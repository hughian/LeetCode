#include<iostream>
using namespace std;
int a[1001];
int b[1001];
class Solution{
public:
    void ReaderNo(){
        for(int i=0;i<1001;i++){
            a[i] = 0;
            b[i] = 0;
        }
        int n;
        cin>>n;
        for(int i=1;i<n+1;i++)
            cin>>a[i];
        for(int i=1;i<=n;i++){
            cout<<++b[a[i]]<<" ";
        }
        cout<<endl;
    }
};
int main()
{
    Solution s;
    s.ReaderNo();
    return 0;
}
