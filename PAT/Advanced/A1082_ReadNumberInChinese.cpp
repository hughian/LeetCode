#include<iostream>
#include<string>
#include<stack>
using namespace std;

string digit[10] = {"ling","yi","er","san","si","wu","liu","qi","ba","jiu"};
string wei[] = {"","Shi","Bai","Qian","Wan","Yi"};

int main()
{
    string str,ans;
    cin>>str;
	int len = str.length();
	int left = 0,right = len-1;
    if(str[0]=='-'){
        ans += "Fu";
		left ++;
    }
	
	while(left +4<=right) right-=4;
	while(left < len){
		bool flag = false;
		bool haveWei = false;
		while(left<=right){
			if(left>0 && str[left]=='0')
				flag = true;
			else{
				if(flag==true){
					ans += (" " + digit[0]);
					flag = false;
				}
				if(left > 0) 
					ans += " ";
				ans += digit[str[left]-'0'];
				haveWei = true;
				if(left != right)
					ans += (" " + wei[right-left]);
			}
			left++;
		}
		if(haveWei && right != len-1)
			ans += (" " + wei[(len-right-1)/4 + 3]);
		right += 4;
	}
    cout<<ans;
    return 0;
}
