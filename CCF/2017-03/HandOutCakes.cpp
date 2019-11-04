#include<iostream>

using namespace std;
int partition(int A[],int low,int high)
{
    int pivot = A[low];
    while(low<high){
        while(pivot<=A[high] && high>low) --high;
        A[low] = A[high];
        while(pivot>=A[low] && low<high) ++low;
        A[high] = A[low];
    }
    A[low] = pivot;
    return low;
}
void QuickSort(int A[],int low,int high)
{
    int pivot;
    if(low<high){
        pivot = partition(A,low,high);
        QuickSort(A,low,pivot-1);
        QuickSort(A,pivot+1,high);
    }
}


int main()
{
    int n,k;
    cin>>n>>k;
    int A[n];
    for(int i=0;i<n;i++){
        cin>>A[i];
    }
    //QuickSort(A,0,n-1);
    int i=0,num = 0,tmp = 0;
    while(i<n){
        tmp += A[i++];
        if(tmp < k ){
            continue;
        }
        num++;
        tmp = 0;
    }
    if(tmp > 0) num++;
    cout<<num<<endl;
    return 0;
}
