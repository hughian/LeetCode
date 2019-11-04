#include<iostream>
using namespace std;
int main()
{
    int i;
    int n;
    cin>>n;
    int tmp;
    int A[n];
    int flg[3]={0,0,0};
    long long sum=0;
    long long edge;
    for(i=0;i<n;i++)
    {
        cin>>tmp;
        A[i]=tmp;
        sum+=tmp;
    }
    if(sum % 4 !=0){
        cout<<"false";
        return 0;
    }
    edge=sum/4;
    i=0;
    while(i<3){
        edge=sum/4;
        for(int j=0;j<n;j++)
        {
            if(A[j]!=-1){
                if(edge-A[j]>0)
                {
                    edge=edge-A[j];
                    A[j]=-1;
                }
                else if(edge-A[j]==0)
                {
                    A[j]=-1;
                    flg[i]=1;
                    break;
                }
                else{
                    cout<<"false";
                    return 0;
                }
            }
        }
        i++;
    }
    if(flg[0]&flg[1]&flg[2])
        cout<<"true";
    return 0;
}

