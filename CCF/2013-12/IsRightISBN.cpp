#include<iostream>

using namespace std;

//  0 - 6 7 0 - 8 2 1 6  2  -  4
//  0 1 2 3 4 5 6 7 8 9 10 11 12
//  1   2 3 4   5 6 7 8  9
int main(void)
{
    char str[14];
    cin>>str;
    int sum = str[0] - '0';
    int k = 2;
    for(int i=2;i<11;i++){
        if(i==5) continue;
        sum += (str[i] - '0') * (k++);
    }
    int check = sum % 11;
	char c = (check==10)?'X':check+'0';
    if(c == str[12]){
        cout<<"Right"<<endl;
    }else{
        str[12] = c;
        cout<<str<<endl;
    }
    return 0;
}
