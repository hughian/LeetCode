#include<iostream>
#include<vector>
using namespace std;

bool win(char f,char s)
{
    if((f == 'C'&& s == 'J') || (f=='J' && s=='B') || (f=='B' && s=='C'))
        return true;
    return false;
}
void count(int a[],char c)
{
    if(c=='B')
        a[0]++;
    else if(c=='C')
        a[1]++;
    else
        a[2]++;
}

void set(char &c,int a[])
{
    if(a[0]>=a[1] && a[0]>=a[2])
        c = 'B';
    else if(a[1]>=a[0] && a[1]>=a[2])
        c = 'C';
    else
        c = 'J';
}

int main()
{
    int n;
    cin>>n;
    char f,s;
    int fw=0,fs=0,sw=0;
    int fbcj[3] ={0};
    int sbcj[3] ={0};
    for(int i=0;i<n;i++){
        cin>>f>>s;
        if(win(f,s)){
            fw ++;
            count(fbcj,f);
        }
        else if(win(s,f)){
            sw ++;
            count(sbcj,s);
        }
        else{ 
            fs ++;
        }
    }
    char rf,rs;
    set(rf,fbcj);
    set(rs,sbcj);

    cout<<fw<<" "<<fs<<" "<<sw<<endl;
    cout<<sw<<" "<<fs<<" "<<fw<<endl;
    cout<<rf<<" "<<rs;
    return 0;
}
