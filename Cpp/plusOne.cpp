#include<iostream>
#include<vector>
using namespace std;

vector<int> plusOne(vector<int>& digits) {
    int len = digits.size();
    int c = 1,tmp;
    for(int i=len-1;i>=0;i--)
    {
        tmp = digits[i]+c;
        digits[i] = tmp%10;
        c = tmp/10;
    }
    if(c!=0)
        digits.insert(digits.begin(),c);
    return digits;
}

int main()
{
    vector<int> v;
    v.push_back(9);
    cout<<v[0]<<endl;
    plusOne(v);
}
