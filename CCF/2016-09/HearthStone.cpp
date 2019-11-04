#include<iostream>
#include<map>
#include<string>
#include<vector>
using namespace std;
#define SUMMON 0
#define ATTACK 1
#define END    2
struct Retinue{
    int attack,health;
    Retinue(int a,int h):attack(a),health(h){}
    Retinue():attack(0),health(0){}
};
class Solution{
    Retinue player[2][8];
    void print(){
        cout<<"============================"<<endl;
        for(int k=0;k<2;k++){
            for(int i=0;i<8;i++){
                cout<<player[k][i].attack<<"  "<<player[k][i].health<<endl;
            }
            cout<<"++++++++++++"<<endl;
        }
    }
public:
    void Hearthstone(){
        int n;
        int flg = 0; //0 是先手玩家
        map<string,int> Trans;
        player[0][0] = Retinue(0,30);
        player[1][0] = Retinue(0,30);
        Trans["summon"] = SUMMON;
        Trans["attack"] = ATTACK;
        Trans["end"]    = END;
        cin>>n;
        string str;
        for(int i=0;i<n;i++){
            cin>>str;
            if(Trans[str] == SUMMON){
                int pos,attack,health;
                cin>>pos>>attack>>health;
                Retinue re(attack,health);
                for(int i=7;i>pos;i--)
                    player[flg][i] = player[flg][i-1];
                player[flg][pos] = re;
            }
            else if(Trans[str] == ATTACK){
                int attacker,defender;
                cin>>attacker>>defender;
                int attack,_attack;
                
                attack = player[flg][attacker].attack;
                _attack = player[1-flg][defender].attack;
                
                player[flg][attacker].health -= _attack;
                if(attacker == 0 && player[flg][0].health <= 0)
                    break;
                if(player[flg][attacker].health <= 0){
                    for(int i=attacker;i<7;i++){
                        player[flg][i] = player[flg][i+1];
                    }
                    player[flg][7] = Retinue();
                }
            
                
                player[1-flg][defender].health -= attack;
                if(defender == 0 && player[1-flg][0].health <= 0)
                    break;
                if(player[1-flg][defender].health <= 0){
                    for(int i= defender;i<7;i++)
                        player[1-flg][i] = player[1-flg][i+1];
                    player[1-flg][7] = Retinue();
                }
            }else{
                flg =1 - flg;
            }
            //print();
        }

        if(player[0][0].health <=0 && player[1][0].health >= 0)
            cout<<-1<<endl;
        else if(player[0][0].health >=0 && player[1][0].health <= 0)
            cout<<1<<endl;
        else
            cout<<0<<endl;

        int tRetinue0 = 0,tRetinue1 = 0;
        for(int i=1;i<8;i++){
            if(player[0][i].health > 0)
                tRetinue0 ++;
            if(player[1][i].health > 0)
                tRetinue1 ++;
        }
        cout<<player[0][0].health<<endl;
        cout<<tRetinue0<<" ";
        for(int i=1;i<8;i++)
            if(player[0][i].health > 0)
                cout<<player[0][i].health<<" ";
        cout<<endl;
        cout<<player[1][0].health<<endl;
        cout<<tRetinue1<<" ";
        for(int i=1;i<8;i++)
            if(player[1][i].health > 0)
                cout<<player[1][i].health<<" ";
    }
};
//#include<stdio.h>
int main()
{
    //freopen("test.txt","r",stdin);
    Solution s;
    s.Hearthstone();
    return 0;
}
