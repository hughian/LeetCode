#include<iostream>
#include<vector>
using namespace std;

    void rotatewithExtraSpace(vector<int>& nums,int k){
        vector<int> v(nums.size());
        int i,j;
        int n = nums.size();
        for(i=0;i<n-k;i++)
            v[i] = nums[i];
		
        for(j=0;i<n;i++,j++)
            nums[j]=nums[i];
        
		for(i=0;i<n-k;i++,j++)
            nums[j]=v[i];
    }
    void reverse(vector<int>& v,int begin,int end){
        int tmp;
        for(int i=begin;i<(begin+end)/2;i++){
            tmp = v[i];
            v[i] = v[begin+end-i];
            v[begin+end-i]=tmp;
        }
    }
    void rotate(vector<int>& nums, int k) {
        if(k>=nums.size())
			return ;
		reverse(nums,0,k);
        reverse(nums,k+1,nums.size()-1);
        reverse(nums,0,nums.size()-1);
    }
	
void print(vector<int> v){
	for(int i=0;i<v.size();i++)
		cout<<v[i]<<" ";
	cout<<endl;
}
int main()
{
	int a[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49};
	vector<int> v(a,a+49);
	print(v);
	rotatewithExtraSpace(v,10);
	print(v);	
	return 0;
}