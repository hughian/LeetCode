#include<iostream>
#include<vector>
#include<cmath>
#include <algorithm> 
using namespace std;
class Solution {
public:
	int partion(vector<int>& nums,int begin=0,int end=0){
		int pivot = nums.at(begin);
		while(begin<end){
			while(begin<end && nums.at(end) >= pivot) --end;
			nums.at(begin) = nums.at(end);
			while(begin<end && nums.at(begin) <= pivot) ++begin;
			nums.at(end) = nums.at(begin);
		}
		nums.at(begin) = pivot;
		return begin;
	}
	bool quicksort(vector<int>& nums,int low,int high){
		if(low<high){
			int pivot = this->partion(nums,low,high);
			this->quicksort(nums,low,pivot-1);
			this->quicksort(nums,pivot+1,high);
		}
	}
    int findPairs(vector<int>& nums, int k) {
        int cnt = 0;
		sort(nums.begin(),nums.end());
		vector<int>::iterator pos = unique(nums.begin(),nums.end());
		nums.erase(pos,nums.end());
		
		int n = nums.size();
        for(int i =0;i<n;i++)
            for(int j =0;j<n;j++){
                if(i != j && abs(nums.at(i) - nums.at(j)) == k){
                        cnt++;
						cout <<"abs("<<nums.at(i)<<" - "<<nums.at(j)<<") = "<<abs(nums.at(i) - nums.at(j))<<endl;
				}
            }
		cout<<cnt/2<<endl;
        return cnt/2;
    }
};

int main()
{
	Solution a;
	int array[] = {3,1,4,1,5};
	vector<int> ar1(array,array+5);
	a.quicksort(ar1,0,ar1.size()-1);
	for(int i =0;i<5;i++)
		cout<<ar1.at(i);
	cout<<endl;
	vector<int> ar2(array,array+5);	

	a.findPairs(ar2,2);
	return 0;
}
