#include<iostream>
#include<vector>
using namespace std;

void reverse(vector<int>& vec,int left,int right)
{
    for(int i=left;i<=(left+right)/2;i++)
    {
        int tmp = vec[i];
        vec[i] = vec[left+right-i];
        vec[left+right-i] = tmp;
    }
}

int main()
{
    vector<int> nums;
    int n,m;
    cin>>n>>m;
    nums.resize(n);
    for(int i=0;i<n;i++)
        cin>>nums[i];
    m = m%n;
    reverse(nums,0,n-m-1);
    reverse(nums,n-m,n-1);
    reverse(nums,0,n-1);
    cout<<nums[0];
    for(int i=1;i<n;i++)
        cout<<" "<<nums[i];
    return 0;
}
