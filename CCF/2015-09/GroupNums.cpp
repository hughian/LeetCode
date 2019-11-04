#include<iostream>
using namespace std;

class Solution{
public:
    void GroupNums(){
        int n,cnt = 0;
        cin>>n;
        int a[n];
        for(int i=0;i<n;i++)
            cin>>a[i];
        int cur = a[0];
        cnt++;
        for(int i=1;i<n;i++){
            if(a[i] != cur){
                cur = a[i];
                cnt ++;
            }
        }
        cout<<cnt<<endl;
    };
};

int main()
{
    Solution s;
    s.GroupNums();
    return 0;
}
