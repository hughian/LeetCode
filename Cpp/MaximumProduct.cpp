#include<stdio.h>
#include<iostream>
#include<vector>
using namespace std;
int maximumProduct(vector<int>& nums) {
    int max[3],min[2];
    int max1,max2;
    int i = 0;
    max[2]=max[1]=max[0]=nums.at(0);    for(i=1;i<3;i++){
        if(nums.at(i) > max[2])
            max[2] = nums.at(i);
        if(nums.at(i) < max[0])
            max[0] = nums.at(i);
        max[1] += nums.at(i);
    }
    max[1] = max[1] - max[0] - max[2];
    min[0] = max[0];
    min[1] = max[1];
    
    while(i<nums.size()){
        if(nums.at(i) > max[2]){
            max[0] = max[1];
            max[1] = max[2];
            max[2] = nums.at(i);
        }
        else if(nums.at(i) > max[1]){
            max[0] = max[1];
            max[1] = nums.at(i);
        }
        else if(nums.at(i) > max[0])
            max[0] = nums.at(i);
        else{
            if(nums.at(i) < min[0]){
                min[1] = min[0];
                min[0] = nums.at(i);
            }
            else if(nums.at(i) < min[1])
                min[1] = nums.at(i);
            else
                ;//do nothing
        }
        i++;
    }
    max1 = max[0] * max[1] * max[2];
    max2 = min[0] * min[1] * max[2];
    return (max1 > max2)?max1:max2;
}
int main()
{
	vector<int> nums;
	nums.push_back(1);
	nums.push_back(2);
	nums.push_back(3);
	nums.push_back(4);
	printf("%d",maximumProduct(nums));
}
