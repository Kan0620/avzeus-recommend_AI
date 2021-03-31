#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import glob

class A2C_zeus():
    
    
    def __init__(self,model,n_step,critic_loss_coef,g,save_name):
        
        self.model=model
        self.n_step=n_step
        self.critic_loss_coef=critic_loss_coef
        self.g=g
        self.save_name=save_name
        self.sup_vec=np.array([0.9,0.7,0.5,0.3])
        
        
        
    def recommend(self,ranked_list=[]):
        
        actress_vec_dir=glob.glob('actress_vecs/*.npy')
        
        rec_vecs=np.zeros(len(actress_vec_dir))
    
        
        all_vec=[]
    
        for one_dir in actress_vec_dir:
            
            vec=np.load(one_dir)
            
            vec=vec/np.linalg.norm(vec)
        
            all_vec.append(vec)
        
        all_vec=np.array(all_vec)
        
        rp_vecs=[]
        
        for group_index in ranked_list:
            
            vec=np.load('rp_vecs/'+str(group_index)+'_rp_vec.npy')
            
            vec=vec/np.linalg.norm(vec)
            
            rp_vecs.append(vec)
            
        rp_vecs=np.array(rp_vecs)
        
        state=[]
        
        for i,now_rp_vec in enumerate(rp_vecs):
            
            for j in range(4-i):
                
                state.append(np.sqrt(np.power(now_rp_vec-rp_vecs[i+j+1],2).sum()))
                
        V,mu,log_var=self.model.predict(np.array(state).reshape(1,-1))
        
        epsilons=np.random.randn(4)*np.sqrt(np.exp(log_var[0]))+mu[0]
        
        sigmoid=lambda x:1/(1+np.exp(-x))
        
        epsilons=sigmoid(epsilons)
        
        epsilons=self.sup_vec*epsilons
        
        epsilons=np.append(1,epsilons)
        
        for one_rp_vec,one_epsilon in zip(rp_vecs,epsilons):
            
            rec_vecs+=one_epsilon*np.dot(one_rp_vec,all_vec.T)
            
            
        with open('actress_id.txt',mode='r') as f:
            
            s=f.read()
            
        s=s.split('\n')
        
        actress_dict={}
        
        for one_info in s:
            
            actress_dict[one_info.split(':')[1]]=int(one_info.split(':')[0])
            
        rec_index=[]
        
        for index in np.argsort(rec_vecs)[::-1][:30]:
            
            name=actress_vec_dir[index].split('/')[-1].split('.')[0]
            
            rec_index.append(actress_dict[name])
        
        return [rec_index,state,epsilons.tolist()]
    
    def learn(self,input_data=[]):
        s=[]
        epsilon=[]
        r=[]
        V_s=[]
        Q=[]
        
        
        
        for one_data in input_data:
            
            s.append(one_data[0])
            epsilon.append(one_data[1])
            r.append(one_data[2])
            
            V,mu,log_var=self.model.predict(np.array(one_data[0]).reshape(1,-1))
            
            V_s.append(V[0][0])
            
        s=np.array(s)
        V_s=np.array(V_s)
        epsilon=np.array(epsilon)
        r=np.array(r)
            
            
        for i in range(len(s)-self.n_step):
            
            q=0
            
            r_s=r[i:i+self.n_step]
            
            for j,one_r in enumerate(r_s):#割引報酬和の計算
                    
                    q+=np.power(self.g,j)*one_r
                                        
            q+=np.power(self.g,self.n_step)*V_s[i+self.n_step]#割引した状態価値を足す
                
                
            Q.append(q)
            
        
            
            
        
        
    def create_train_data(self):
        
        s=[]
        V_s=[]
        actions=[]
        r=[]
        Q=[]
        
        for i in range(self.n_count):#n_count回探索
            
            done=False
            
            obs=self.env.reset()
            
            while not done:
                
                s.append(obs)#状態sを保存
                
                V,mu,var=self.model.predict(obs.reshape((1,-1)))
                #状態価値と平均,分散取得
                
                
                V=V[0][0]#V(s)
                
                V_s.append(V)
                
                
                
                action=np.random.randn(self.n_action)*np.sqrt(np.exp(var[0]))+mu[0]
                
                
                obs,reward,done=self.env.step(action.tolist())
                
                actions.append(action)#行動のindexを保存
                
                r.append(reward)#即時報酬保存
                
                
        s=np.array(s)
        V_s=np.array(V_s)
        actions=np.array(actions)
        r=np.array(r)
        
        for i in range(len(s)-self.n_step):#n_step行動価値計算
            
            q=0
            
            r_s=r[i:i+self.n_step]
            
            if any([one_game_over_r in r_s for one_game_over_r in self.game_over_r]):
                #n_stepの間にgameoverしている場合
                
                game_over_flag=0
                count=0
                
                while game_over_flag==0:#game_overのrまで割引報酬を足す
                    
                    now_r=r_s[count]
                    
                    q+=np.power(self.g,count)*now_r
                    
                    if now_r in self.game_over_r:
                        
                        game_over_flag=1
                        
                    count+=1
                
            else:
                
                for j,one_r in enumerate(r_s):#割引報酬和の計算
                    
                    q+=np.power(self.g,j)*one_r
                                        
                q+=np.power(self.g,self.n_step)*V_s[i+self.n_step]#w割引した状態価値を足す
                
                
            Q.append(q)
        
        Q=np.array(Q)
        
                
        return s[:len(s)-self.n_step],actions[:len(s)-self.n_step],Q
    
    def test(self):
        
        r=0
        
        for i in range(self.n_test):
            
            done=False
            
            obs=self.env.reset()
            
            
            while not done:
                
                V,mu,var=self.model.predict(obs.reshape((1,-1)))
                
                action=np.random.randn(self.n_action)*np.sqrt(np.exp(var[0]))+mu[0]
                
                obs,reward,done=self.env.step(action.tolist())
                
                r+=reward
                
            
                
        return r/self.n_test
    
    def fit(self):
        
        epoch=1
        score=self.finish_score-1
        
        while score<self.finish_score:
            
            s,actions,Q=self.create_train_data()
            
            
            self.model.fit(s,actions,Q,self.critic_loss_coef)
            
            score=self.test()
            
            self.model.save(self.save_name)
            
            print('epoch:'+str(epoch)+'  score:'+str(score))
            print('')
            
            epoch+=1
            
            
            
            
################ cart_poleでの使用例 ##############
        
import gym

class environment():
    
    def __init__(self):
        
        self.env=gym.make('MountainCarContinuous-v0')
        
        self.env.reset()
        
        self.count=0
        
    def reset(self):
        
        self.count=0
        
        return self.env.reset()
    
    def step(self,action=[]):
        
        observation,reward,done,info=self.env.step(action)
        
        
        
        self.count+=1
        
       
        if done:
            
            if  self.count==999:
                
                reward=-1
                
            else:
                print('clear',self.count)
                reward=1
            
            
        else:
            
            reward=0
            
            
        
            
        return observation,reward,done

import torch
from torch import tensor
from torch.nn import functional as F,Linear,Module,utils
from torch.optim import Adam


        
class model():
    
    def __init__(self):
        
        class Net(Module):
            
            def __init__(self):
                      
                super(Net,self).__init__()
                self.fc1=Linear(10,32)
                self.fc2=Linear(32,32)
                self.critic=Linear(32,1)
                self.actor_mu=Linear(32,4)
                self.actor_log_var=Linear(32,4)
                 
                
                  
        
            def forward(self,x):
        
                h1=F.relu(self.fc1(x))
                h2=F.relu(self.fc2(h1))
                V=self.critic(h2)
                mu=self.actor_mu(h2)
                log_var=self.actor_log_var(h2)
        
                return V,mu,log_var
        
        
       
        self.net=Net()
        print(self.net)
        self.optim=Adam(self.net.parameters(),lr=0.01)
       
        
    def fit(self,s,actions,Q,critic_loss_coef):
        
        self.net.train()
        
        
        s=tensor(s,dtype=float)
        actions=tensor(actions,dtype=float)
        Q=tensor(Q,dtype=float)
        
        output_V,output_mu,output_log_var=self.net(s.float())#V,π取得
        
        log_prob=(-(output_log_var+(actions-output_mu).pow(2)/output_log_var.exp())/2).sum(axis=1)
        #ガウス方策より
        
        
        adv=Q-output_V.view(-1)#アドバンテージ関数取得
        
        actor_loss=-(adv.detach()*log_prob).mean()#方策勾配定理よりactorのloss計算
         
        critic_loss=critic_loss_coef*adv.pow(2).mean()#二乗誤差からcriticのloss計算
        
        total_loss=actor_loss+critic_loss
        
        print(-actor_loss.detach().numpy(),critic_loss.detach().numpy())
        
        self.optim.zero_grad()
        total_loss.backward()
        utils.clip_grad_norm(self.net.parameters(),0.5)#更新を抑える
        self.optim.step()
        
        
            
    def predict(self,x):
        
        self.net.eval()
        
        with torch.no_grad():
        
            x=tensor(x,dtype=float)
        
            V,mu,var=self.net(x.float())
        
        return V.detach().numpy(),mu.detach().numpy(),var.detach().numpy()
    
    
    def save(self,name=str()):
        
        torch.save(self.net.state_dict(),name)
        
        
        
        
        
        
def test_rec(ranked_list=[]):
    
    zeus=A2C_zeus(model(),2,0.4,0.99,'zeus')
    
    
    return zeus.recommend(ranked_list)










