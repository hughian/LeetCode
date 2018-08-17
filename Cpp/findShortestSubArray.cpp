#include<iostream>
#include<vector>
#define INT_MAX 2147483647
using namespace std;
class Solution {
public:
    int findShortestSubArray(vector<int>& nums) {
        vector<int> tag(50000);
        vector<int> v;
        int max = 0;
        for(int i=0;i<nums.size();i++){
            tag[nums[i]] ++;
            if(tag[nums[i]]>max){
                max = tag[nums[i]];
            }
        }
        for(int i=0;i<50000;i++)
            if(tag[i] == max)
                v.push_back(i);
			
		
		
        int min = INT_MAX;
        int tmp = 0;
        for(int i=0;i<v.size();i++){
			cout<<"#"<<v[i]<<endl;
			int frq = v[i];
            int m=0,n=nums.size();
            while(nums[m]!=frq) m++;
            while(nums[n]!=frq) n--;
            tmp = n-m+1;
            if(tmp < min) min = tmp;
        }
        return min; 
    }
};

int main()
{
	int array[] = {1,3,2,2,2,1,1,3,1,1,2};
	vector<int> nums(array,array+sizeof(array)/sizeof(int));
	Solution a;
	cout<<a.findShortestSubArray(nums)<<endl;
	return 0;
}