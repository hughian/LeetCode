#include<iostream>
#include<vector>
using namespace std;

class Solution{
public:
    vector<int> TwoSum(vector<int> v,int target)
    {
        int tmp;
        int i=0;
        int size=v.size();
        vector<int> a(0);
        while(i < size){
            tmp=target-v[i];
            for(int j=i+1;j<size;j++){
                if(tmp==v[j])
                {   
                    a.push_back(v[i]);
                    a.push_back(v[j]);
                    return a;
                }
            }
            i++;
        }
    }
};

int main()
{
    Solution a;
    vector<int> t(0);
    t.push_back(3);
    t.push_back(2);
    t.push_back(4);
    vector<int> res=a.TwoSum(t,6);
    cout<<res[0]<<" "<<res[1];
    return 0;
}
