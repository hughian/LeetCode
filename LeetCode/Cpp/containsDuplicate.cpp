#include<iostream>
#include<vector>
using namespace std;
	
	int partion(vector<int>& nums,int low,int high){
        int pivot = nums[low];
        while(low<high){
            while(nums[high]>pivot && high > low) high--;
            nums[low] = nums[high];
            while(nums[low]<pivot && low<high) low++;
            nums[high] = nums[low];
        }
        nums[low] = pivot;
        return pivot;
    }
    void QuickSort(vector<int>& nums,int low,int high){
        if(low<high){
            int pivot = partion(nums,low,high);
            QuickSort(nums,low,pivot-1);
            QuickSort(nums,pivot+1,high);
        }
    }
    bool containsDuplicate(vector<int>& nums) {
        if(nums.size()<2) return false;
        QuickSort(nums,0,nums.size()-1);
        for(int i=0;i<nums.size()-1;i++)
        {
            if(nums[i] == nums[i+1])
                return true;
        }
        return false;
    }
	
int main()
{
	vector<int> v;
	v.push_back(3);
	v.push_back(3);
	cout<<containsDuplicate(v)<<endl;
	return 0;
}