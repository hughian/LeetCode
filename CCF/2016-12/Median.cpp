#include<iostream>
#include<vector>
/*
问题描述
　　在一个整数序列a1, a2, …, an中，如果存在某个数，大于它的整数数量等于小于它的整数数量，则称其为中间数。在一个序列中，可能存在多个下标不相同的中间数，这些中间数的值是相同的。
　　给定一个整数序列，请找出这个整数序列的中间数的值。
输入格式
　　输入的第一行包含了一个整数n，表示整数序列中数的个数。
　　第二行包含n个正整数，依次表示a1, a2, …, an。
输出格式
　　如果约定序列的中间数存在，则输出中间数的值，否则输出-1表示不存在中间数。
 */
using namespace std;
int partion(vector<int>& vec,int low,int high){
	int pivot = vec[low];
	while(low<high){
		while(vec[high] >= pivot && high>0) --high;
		vec[low] = vec[high];
		while(vec[low] <= pivot && low < high) ++low;
		vec[high] = vec[low];
	}
	vec[low] = pivot;
	return low;
}
int sort(vector<int>& vec,int low,int high){
	if(low<high){
		int pivot = partion(vec,low,high);
		sort(vec,low,pivot-1);
		sort(vec,pivot+1,high);
	}
}

int main(int argc,char *argv[])
{
    int n,tmp;
    int mid=0,cnt=0;
    cin>>n;
	vector<int> nums(n,0);
    for(int i =0;i<n;i++){
		cin>>tmp;
		nums[i] = tmp;
    }
	sort(nums,0,n-1);
	mid = n / 2;
    int i = mid;
    while(i<n && nums[i]==nums[mid]) { cnt++; i++;}
    i = mid - 1;
    while(i>=0 && nums[i]==nums[mid]) { cnt++; i--;}
    if((cnt+n)%2 != 0)
      cout<<-1;
    else
      cout<<nums[mid];
    return 0;
}
