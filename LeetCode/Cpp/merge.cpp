#include<iostream>
#include<vector>
using namespace std;

    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i=n,j=0;
        for(int k=m-1;k>=0;k--)
            nums1[n+k]=nums1[k];
        int k=0;
        while(i<n+m && j<n){
            if(nums1[i]<nums2[j])
                nums1[k++] = nums1[i++];
            else
                nums1[k++] = nums2[j++];
        }
        while(j<n){
            nums1[k++] = nums2[j++];
        }

        for(i=0;i<m+n;i++)
            cout<<nums1[i]<<" ";
    }

int main()
{
    vector<int> nums1;
    vector<int> nums2;
    nums1.push_back(1);
    nums1.push_back(0);
    nums2.push_back(2);
    merge(nums1,1,nums2,1);
    return 0;
}
