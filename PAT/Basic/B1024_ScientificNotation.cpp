#include<iostream>
#include<string>
#include<cstdlib>
using namespace std;
void swap(string &str)
{
    char t = str[0];
    str[0] = str[1];
    str[1] = t;
}

int main()
{
    string str;
    cin>>str;

    int r = str.find("E");
    string e = str.substr(1,r-1);
    string x = str.substr(r+1,str.length()-r-1);
    int xi = atoi(x.c_str());
    char t;
    if(str[0]=='-')
        cout<<"-";
    
    if(xi == 0)
        cout<<e;
    else if (xi<0){
        swap(e);
        xi++;
        while(xi<0){
            e[0] = '0';
            e = "." + e;
            xi++;
        }
        e = "0" + e;
        cout<<e;
    }else{
        if(xi>=e.length()-2){
            xi -= (e.length()-2);
            swap(e);
            e = e.substr(1,e.length()-1);
            while(xi>0){
                e += "0";
                xi--;
            }
            cout<<e;
        }
        else{
            int k=1;
            for(;k<=xi;k++){
                t = e[k];
                e[k] = e[k+1];
                e[k+1] = t;
            }
            cout<<e;
        }
    }
    return 0;
}
