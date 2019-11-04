#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

//reverse(str.begin(),str.end());
string add(string A,string B)
{
    string C;
    char ch;
    int r = 0,tmp;
    for(int i=0;i<(int)A.length();i++){
        tmp = A[i]-'0' + B[i] - '0' + r;
        r = tmp/10;
        ch = tmp % 10 + '0';
        C.push_back(ch);
    }
    if(r!=0){
        ch = r +'0';
        C.push_back(ch);
    }
    reverse(C.begin(),C.end());
    return C;
}
bool isParlindrome(string str)
{
    string tmp = str;
    reverse(tmp.begin(),tmp.end());
    return tmp==str;
}
int main()
{
    string A;
    cin>>A;
    string C = A;
    int cnt = 0,flg = 0;
    while(!isParlindrome(C)){
        string B = A;
        reverse(B.begin(),B.end());
        C = add(A,B);
        cout<<A<<" + "<<B<<" = "<<C<<endl;
        A = C;
        cnt++;
        if(cnt>=10){
            flg = 1;break;
        }
    }
    //输出要严格一致，注意标点 . 
    if(flg)
        cout<<"Not found in 10 iterations.";
    else
        cout<<A<<" is a palindromic number.";

    return 0;
}
