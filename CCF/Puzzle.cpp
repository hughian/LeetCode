#include<iostream>
#include<cstdio>
using namespace std;

class Solution{
public:
    void Puzzle(){
        int n,m;
        cin>>n>>m;
        long long  k = n/3;
        cout<<(2*k)%1000000007;
    }
};

int main()
{
    Solution s;
    s.Puzzle();
    return 0;
}
