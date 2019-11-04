#include<iostream>
#include<string>
using namespace std;
int finde(string str){
    int res = 0;
    string::size_type r = str.find('.');
    int Dot = (r == string::npos )?str.size():r;
    string::size_type FirstDigit = str.find_first_not_of("0.");
    if(FirstDigit == string::npos)
        return 0;
    else{
        res = Dot-FirstDigit;
        return res>0?res:res+1; //小数点出现在有效位数的前面是应该加一
    }
}

string getstd(const string &str,int n){
    int FirstDigit = str.find_first_not_of("0.");
    string tmp;
    string::size_type i = FirstDigit ;
    while(n && i<str.length()){
        if(str[i] != '.'){
            tmp.push_back(str[i]);
            n--;
        }
        i++;
    }
    while(n){
        tmp.push_back('0');
        n--;
    }
    return tmp;
}
string getres(const string &str,int n){
	string ans = getstd(str,n);
	if(ans=="0")
		return "0.0"; //0.0 情况单独判断
	else
		return "0."+ans+"*10^" + to_string(finde(str));
}

int main()
{
    string a,b;
    int n;
    cin>>n>>a>>b;
    string as = getres(a,n);
    string bs = getres(b,n);
    if(as == bs)
        cout<<"YES "<<as<<endl;
    else
        cout<<"NO "+ as + " "+bs<<endl;
    return 0;
}
