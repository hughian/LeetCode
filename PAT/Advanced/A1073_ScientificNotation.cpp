#include<iostream>
#include<string>
using namespace std;

int main()
{
    string str,res;
    cin>>str;
    int r = str.find('E');
    string sign = str[0]=='+'?"":"-";
    string as = str.substr(1,r-1);
    string es = str.substr(r+1,str.length()-r-1);
    //cout<<as<<endl<<es<<endl;
    int exp = stoi(es);
    if(exp>0){
        int i=1;
        while(i<(int)as.length()-1 && exp>0){
            char t = as[i];
            as[i] = as[i+1];
            as[i+1] = t;
            i++;exp--;
        }
        if(i==(int)as.length()-1 && exp == 0){
            as = as.substr(0,i);
        }else if(exp > 0){
            as[i] = '0';
            exp--;
            while(exp>0){
                as+="0";exp--;
            }
        }
        //cout<<as;
    }else if(exp<0){
        while(exp<0){
            char t = as[0];
            as[0] = as[1];
            as[1] = t;
            as = "0" + as;
            exp++;
        }
    }
    cout<<(sign+as);
    return 0;
}
