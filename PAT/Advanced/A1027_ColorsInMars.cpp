#include<iostream>
#include<string>
using namespace std;

string toNum(int x){
    string ans;
    int low = x % 13;
    int high = x/13;
    if(high<10)
        ans.push_back(high+'0');
    else
        ans.push_back(high-10+'A');
    if(low < 10)
        ans.push_back(low+'0');
    else
        ans.push_back(low-10+'A');
        return ans;
}

int main()
{
    int r,g,b;
    cin>>r>>g>>b;
    string ans;
    ans = "#" + toNum(r)+toNum(g)+toNum(b);
    cout<<ans;
    return 0;
}
