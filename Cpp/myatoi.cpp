#include<iostream>
using namespace std;

class Solution {
public:
    int myAtoi(string str) {
        int k,i;
        double r=0.0;
        const double max=2147483647;
        int sgn=0;
        if(str.empty())
            return 0;
        int size=str.size();
        for(k=0;k<size;k++){ 
            if(str[k]==' ')
                continue;
            else if(str[k]=='+' || str[k]=='-'){
                if(str[k]=='-')
                    sgn=1;
                for(i=k+1;i<size;i++){
                    if(str[i]>='0' && str[i]<='9')
                        r = r * 10 + str[i] - '0';
                    else
                        break;
                }
            }
            else{
                for(i=k;i<size;i++){
                    if(str[i]>='0' && str[i] <= '9')
                        r = r * 10 + str[i] - '0';
                    else
                        break;
                }
            }
            if(sgn)
                return -r;
            return r>max?max:r;
        }
        return 0;
    }
};


int main()
{
	string str="2147483648";
	Solution at;
	cout<<at.myAtoi(str)<<endl;
	return 0;
}
