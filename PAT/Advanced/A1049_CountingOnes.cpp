#include<iostream>
#include<vector>
using namespace std;
int main()
{
    int N;
    cin>>N;
    int left,right,now,a=1,ans=0;
    while(N/a){
        left = N/(a*10); //left 表示N的左边部分
        right = N % a;   //right 表示N的右边部分
        now = N / a % 10; //now 表示N的当前位
		/*
			N  = |dn dn-1 dn-2 ...| di  |... d2 d1 d0 |
			     |                |  ^  |             |
				 |	<-- left -->  | now |<-- right -->|
		*/
		//当前位置为0时，左边left-1时会出现1，共left次，又会重复a（0...0~9...9次） 共 left * a
        if(now == 0) ans+= left * a;   
         //当前位置为1时，左边出现left次，重复a次共 left*a，当前位置为1的数右边共有right个，还有一个当前位置的1
		else if(now ==1) ans += left*a + right + 1;
        //当前位置大于1时，左边出现left次，重复a次共left*a，now位置大于1，会出现1，计a次 
		else ans += (left + 1) * a; 
        a = a*10; 
    }
    cout<<ans;
    return 0;
}
