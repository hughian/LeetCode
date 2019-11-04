#include<iostream>
#include<vector>
#include<cstring>
#include<cstdio>
using namespace std;
vector<double> nums;
double str2num(char buf[])
{
    bool point = false,flag = true;
    double res = 0;
    int sign = 1;
    double r = 0.1;
    unsigned int i = 0;
    if(buf[0]=='-'){
        sign = -1;
        i++;
    }
    for(;i<strlen(buf);i++){
        if(buf[i]>='0' && buf[i]<='9'){
            if(point==false)
                res = res*10 + buf[i] - '0';
            else{
                if(r<0.01){
                    flag = false;break;
                }               
                res = res + r * (buf[i] - '0');
                r = r * 0.1;
            }
        }else if(buf[i]=='.' && point == false){
            point = true;
        }else{
            flag = false;break;
        }
    }
    if(res >1000 || res < -1000)
        flag = false;
    if(!flag)
        printf("ERROR: %s is not a legal number\n",buf);
    else
        nums.push_back(res*sign);
}
int main()
{
    int n;
    cin>>n;
    char buf[100];
    double d;
    for(int i=0;i<n;i++){
        scanf("%s",buf);
        str2num(buf);
    }
    if(nums.size()==0)
        printf("The average of 0 numbers is Undefined");
    else if(nums.size()==1)
        printf("The average of 1 number is %.2lf",nums[0]);
    else{
        double sum = 0;
        for(unsigned i=0;i<nums.size();i++){
            sum += nums[i];
        }
        printf("The average of %d numbers is %.2lf",nums.size(),sum/nums.size());
    }
    return 0;
}
