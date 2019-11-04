#include<iostream>
#include<string>
#include<stack>
using namespace std;
string digits[10] = {"zero","one","two","three","four","five","six","seven","eight","nine"};
int main()
{
    long long sum = 0;
    string n;
    cin>>n;
    for(int i=0;i<(int)n.length();i++)
    {
        sum += n[i] -'0';
    }
    string str ="";
    if(sum == 0){
        str = digits[sum];
        cout<<str;
    }else{
        while(sum){
            str = " " + digits[sum%10] + str;
            sum = sum/10;
        }
        cout<<str.substr(1,str.size()-1);
    }
    return 0;
}
