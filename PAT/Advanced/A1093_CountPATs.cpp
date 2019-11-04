#include<iostream>
#include<string>
#include<vector>
using namespace std;

int main()
{
    string str;
    cin>>str;
    int len = str.length();
	vector<int> ps(len+1,0),ts(len+1,0);
    long long res= 0;
	//使用穷举可以得15分，后三个测试点会超时
    /*for(int i=0;i<len;i++){
        for(int j=i+1;j<len;j++){
            for(int k=j+1;k<len;k++){
                if(str[i]=='P' && str[j]=='A' && str[k]=='T')
                    res++;
            }
        }
    }*/
	
	ps[0] = (str[0]=='P')?1:0;
    ts[0] = (str[0]=='T')?1:0;
    for(int i=1;i<len;i++){
        ps[i]=ps[i-1];
        ts[i]=ts[i-1];
        if(str[i]=='P')	ps[i]++;
        if(str[i]=='T')	ts[i]++;
    }
    for(int i=1;i<len;i++){
        if(str[i]=='A')
            res += (ps[i-1] * (ts[len-1]-ts[i]));
    }
    cout<<(res%1000000007);
    return 0;
}
