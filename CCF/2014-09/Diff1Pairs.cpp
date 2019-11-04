#include<iostream>
using namespace std;
class Solution{
public:
    int Partition(int a[],int low,int high){
        int pivot = a[low];
        while(low<high){
            while(high > low && a[high] >= pivot) --high;
            a[low] = a[high];
            while(low < high && a[low] <= pivot) ++low;
            a[high] = a[low];
        }
        a[low] = pivot;
        return low;
    }
    void QuickSort(int a[],int low,int high){
        if(low<high){
            int pivot = Partition(a,low,high);
            QuickSort(a,low,pivot-1);
            QuickSort(a,pivot+1,high);
        }
    }
    void DiffOnePairs(){
        int n;
        int cnt = 0;
        cin>>n;
        int a[n];
        for(int i=0;i<n;i++)
            cin>>a[i];
        QuickSort(a,0,n-1);
        for(int i=0;i<n-1;i++){
            if(a[i+1] - a[i] == 1)
                cnt ++;
        }
        cout<<cnt<<endl;
    }
};
int main()
{
    Solution s;
    s.DiffOnePairs();
    return 0;
}
