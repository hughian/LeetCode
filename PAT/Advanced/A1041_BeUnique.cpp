#include<iostream>
#include<vector>
using namespace std;
vector<int>  nums(10010,0);
vector<int>  seq;
int main()
{
    int n,t;
    cin>>n;
    for(int i=0;i<n;i++){
        cin>>t;
        if(nums[t]==0){
            seq.push_back(t);
        }
        nums[t]++;
    }
    for(unsigned i=0;i<seq.size();i++){
        if(nums[seq[i]]==1){
            cout<<seq[i];
            return 0;
        }
    }
    cout<<"None";
    return 0;
}
