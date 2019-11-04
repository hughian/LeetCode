#include<iostream>
#include<string>
#include<vector>
using namespace std;
int arr[37] = {0};
int getindex(char c)
{
    if(c>='0' && c<='9')
        return c-'0';
    if(c>='a' && c<='z')
        return c-'a'+10;
    if(c=='_')
        return 36;

}
char getch(int x)
{
    if(x>=0 && x<=9)
        return x+'0';
    if(x>=10 && x<=35)
        return x-10+'a';
    if(x==36)
        return '_';
}
bool ansfind(vector<char>& ans,char c)
{
    for(unsigned i=0;i<ans.size();i++)
        if(ans[i] == c)
            return true;
    return false;
}

int main()
{
    int k;
    string str;
    cin>>k>>str; //str长度至少为k
    vector<char> ans;
    int i=0,len = str.length();
    while(i<len){    
        int idx = getindex(str[i]);
        if(arr[idx]==-1){
            i++;continue;
        }
        int j=i;
        for(;j<i+k && j<len;j++){
            if(str[j] != str[i])
                break;
        }
        if(j<i+k)
            arr[idx] = -1;
        else{ 
            if(arr[idx]!=-1){
                arr[idx] = 1;
                if(!ansfind(ans,str[i]))
                    ans.push_back(str[i]);
            }
        }
        i = j;
    }

    for(unsigned i=0;i<ans.size();i++){
        if(arr[getindex(ans[i])]==1)
            cout<<ans[i];
    }
    cout<<endl;
    i=0;
    while(i<len){
        cout<<str[i++];
        if(arr[getindex(str[i])]==1){
            i+=(k-1);
        }
    }
    return 0;
}
