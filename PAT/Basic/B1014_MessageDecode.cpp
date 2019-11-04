#include<iostream>
#include<string>
#include<cstdio>
using namespace std;
string DAY[7]={"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"}; 

int main()
{
    string s1,s2,s3,s4;
    cin>>s1>>s2>>s3>>s4;
	
    int d=0,h=0,m=0;
    unsigned int i=0;
    for(; i < s1.length() && i < s2.length();i++){
        if(s1[i]==s2[i] && s1[i] >= 'A' && s1[i] <= 'G'){
            d = s1[i]-'A';
            break;
        }
    }
    for(i++;i<s1.length() && i<s2.length();i++){
        if(s1[i]==s2[i]){
            if(s1[i] >= '0' && s1[i] <= '9'){
                h = s1[i]-'0';break;
            }else if( s1[i] >='A' && s1[i]<='N'){
                h =(s1[i]-'A') + 10;break;
            }
        }
    }
    for(unsigned int i=0;i<s3.length() && i<s4.length();i++){
        if(s3[i] == s4[i]){
            if((s3[i]>='A' && s3[i]<='Z') || (s3[i]>='a' && s3[i]<='z')){
                m = i;
                break;
            }
        }
    }
	printf("%s %02d:%02d",DAY[d].data(),h,m);
    return 0;
}
