#include<iostream>
#include<iomanip>

using namespace std;

class A{
    int n;
    int index;
public:
    int *a;
    A(int num);
    A& add(int e);
    int max_subarray();
};
A::A(int num):n(num),index(0),a(new int[num])
{
    for(int i=0;i<num;i++)
        a[i]=0;
}
A& A::add(int e)
{
    if(index<n)
        a[index++]=e;
    else
        cout<<"Full"<<endl;
}
int A::max_subarray()
{
    int maxend;
    int maxsof;
    int i;
    int low,high,mid;
    low=high=mid=0;
    maxend=maxsof=a[0];
    for(i=1;i<n;i++)
    {
        if(a[i]+maxend >= a[i])
            maxend+=a[i];
        else{
            maxend=a[i];
            low=i;
        }
        if(maxsof < maxend){
            maxsof=maxend;
            high=i;
        }
        cout<<setw(3)<<maxend<<setw(6)<<maxsof<<setw(6)<<low<<setw(3)<<high<<endl;
    }
    return maxsof;
}

int main()
{
    A a(9);
    a.add(-2);
    a.add(1);
    a.add(-3);
    a.add(4);
    a.add(-1);
    a.add(2);
    a.add(1);
    a.add(-5);
    a.add(4);
    cout<<endl<<a.max_subarray()<<endl;
    return 0;
}
