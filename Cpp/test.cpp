#include <iostream>
#include "findMedianSortedArrays.cpp"
using namespace std;
vector<int> va,vb;
void init(){
    int m,n;
    int tmp;
    cin>>m;
    for(int i=0;i<m;i++)
    {
        cin>>tmp;
        va.push_back(tmp);
    }
    cin>>n;
    for(int i=0;i<n;i++)
    {
        cin>>tmp;
        vb.push_back(tmp);
    }
}


int main(){
    Solution a;
    init();
    cout<<a.findMedianSortedArrays(va,vb);
    return 0;
}
