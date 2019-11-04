#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
//B1019
string fill(string &str){
    while(str.length()<4){
        str = "0"+str;
    }
    return str;
}

int main()
{
    string str;
    cin>>str;
    fill(str);
    sort(str.begin(),str.end());
    string min=str,max=str;
    reverse(max.begin(),max.end());
    int maxi = stoi(max);
    int mini = stoi(min);
    int diff = maxi - mini;
    string difs = to_string(diff);
    if(diff==0){
        cout<<( max+" - "+min+" = "+fill(difs) )<<endl;
        return 0;
    }
    while(diff!=6174){
        cout<<( max+" - "+min+" = "+fill(difs) )<<endl;
        min = difs;
        sort(min.begin(),min.end());
        max = min;
        reverse(max.begin(),max.end());
        maxi = stoi(max);
        mini = stoi(min);
        diff = maxi-mini;
        difs = to_string(diff);
    }
    cout<<( max+" - "+min+" = "+fill(difs) )<<endl;
    return 0;
}
