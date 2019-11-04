#include<iostream>
#include<string>
using namespace std;

int w[17] = {7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2};
char m[11] = {'1','0','X','9','8','7','6','5','4','3','2'};
int main()
{
    string str;
    int n;
    cin>>n;
    int flg = 0;
    for(int i=0;i<n;i++){
        cin>>str;
        int sum = 0;
        for(int j=0;j<17;j++)
            sum += ((str[j] - '0')*w[j]);
        sum = sum%11;
        if(str[17]!=m[sum]){
            cout<<str<<endl;
            flg = 1;
        }
    }
    if(!flg)
        cout<<"All passed";
    return 0;
}
