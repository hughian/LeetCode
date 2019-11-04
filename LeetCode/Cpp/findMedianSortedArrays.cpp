#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        double ans=0.0;
        int maxleft;
        int minright;
        unsigned int m,n;
        int i,j;
        int imin,imax;
        m = nums1.size();
        n = nums2.size();
        i = 0;
        if(n>=m){
            imin = 0; imax = m;
            while(imin <= imax){
                i = (imin + imax)/2;
                j = (m + n + 1)/2 - i;
                if(j>0 && nums2[j-1]>nums1[i]){
                    imin = i+1;
                }
                else if(i < m && nums1[i-1] > nums2[j]){
                    imax = i-1;
                }
                else//((j==0 || i==m || nums2[j-1] <= nums1[i]) && (i==0 || j==n || nums1[i-1] <= nums2[j])
                {
                    if(j==0)
                        maxleft = nums1[i-1];
                    else if(i==0)
                        maxleft = nums2[j-1];
                    else
                        maxleft = max(nums1[i-1],nums2[j-1]);
                    break;
                }
            }
        } else {
            imin = 0; imin = n;
            while(imin <= imax){
                i = (imin+imax)/2;
                j = (m+n+1)/2 - i;
                if(nums1[j-1] <= nums2[i] && nums2[i-1] <= nums1[j]){
                    break;
                }
                else if(nums1[j-1] > nums2[i]){
                    imin = i+1;
                }
                else{ //nums2[i-1] > nums1[j]
                    imax = i-1;
                }
            }
        }
        if((m+n)%2==0){
            ans = (maxleft + min(nums1[i],nums2[j]))/2.0;
        }
        else{
            ans = maxleft;
        }
        return ans;
    }
};
