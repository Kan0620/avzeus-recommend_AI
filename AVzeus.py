#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import glob
from facenet_pytorch import MTCNN,InceptionResnetV1
from PIL import Image
import cv2




class A2C_zeus():


    def __init__(self,model,load_name,n_step,critic_loss_coef,g,save_name):

        self.model=model
        self.model.load(load_name)
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
            
            if len(one_info)!=0:

                actress_dict[one_info.split(':')[1]]=int(one_info.split(':')[0])

        rec_index=[]

        for index in np.argsort(rec_vecs)[::-1][:30]:

            name=actress_vec_dir[index].split('/')[-1].split('.')[0]

            rec_index.append(actress_dict[name])

        epsilons=epsilons.tolist()

        epsilons.pop(0)

        return [rec_index,state,epsilons]
    
    def img_path2ranked_list(self,load_dir,save_dir):
        
        img=Image.open(load_dir)
        
        mtcnn=MTCNN(image_size=160, margin=10)
    
        img=mtcnn(img,save_dir)
        
        if img==None:
            
            return img
        
        else:
            
            resnet=InceptionResnetV1(pretrained='vggface2').eval()
            
            img = cv2.imread(save_dir)
        
            img=img.transpose(2,0,1)
            
            img=tensor(img,dtype=float)
            
            img=img.reshape(1,3,160,160)
            
            vec=resnet(img.float()).detach().numpy()
            
            vec=vec/np.linalg.norm(vec)
            
            rp_vecs=[]
            
            for i in range(9):
                
                one_vec=np.load('rp_vecs/'+str(i)+'_rp_vec.npy')

                one_vec=one_vec/np.linalg.norm(one_vec)

                rp_vecs.append(one_vec)
                
            rp_vecs=np.array(rp_vecs)
            
            rp_vecs=rp_vecs.reshape(9,512)
            
                
            product=np.dot(rp_vecs,vec.T).reshape(9)
            
            return np.argsort(product).tolist()[::-1][:5]
            
            
        
        
        

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
        
        if len(s)<3:
            
            return 0


        for i in range(len(s)-self.n_step):

            q=0

            r_s=r[i:i+self.n_step]

            for j,one_r in enumerate(r_s):#割引報酬和の計算

                    q+=np.power(self.g,j)*one_r

            q+=np.power(self.g,self.n_step)*V_s[i+self.n_step]#割引した状態価値を足す


            Q.append(q)

        self.model.fit(s[:len(s)-self.n_step],epsilon[:len(s)-self.n_step],Q,self.sup_vec,self.critic_loss_coef)

        self.save()

        return 1

    def manual_teach(self):



        ranked_list=[int(i) for i in input('好きな順を,で区切って入力').split(',')]

        index,state,epsilon=self.recommend(ranked_list)

        print(index)

        r=-(float(input('何位？'))-1.0)/10.0

        return [state,epsilon,r]





    def save(self):

        self.model.save(self.save_name)





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

        self.optim=Adam(self.net.parameters(),lr=0.01)


    def fit(self,s,epsilon,Q,sup_vec,critic_loss_coef):

        self.net.train()


        s=tensor(s,dtype=float)
        epsilon=tensor(epsilon,dtype=float)
        Q=tensor(Q,dtype=float)
        inverce_sup_vec=1/tensor(sup_vec,dtype=float).expand(len(epsilon),4)

        output_V,output_mu,output_log_var=self.net(s.float())

        y=epsilon*inverce_sup_vec



        log_prob=(-(output_log_var+((y/(1-y)).log()-output_mu).pow(2)/output_log_var.exp())/2).sum(axis=1)
        #ガウス方策より


        adv=Q-output_V.view(-1)#アドバンテージ関数取得

        actor_loss=-(adv.detach()*log_prob).mean()#方策勾配定理よりactorのloss計算

        critic_loss=critic_loss_coef*adv.pow(2).mean()#二乗誤差からcriticのloss計算

        total_loss=actor_loss+critic_loss


        self.optim.zero_grad()
        total_loss.backward()
        utils.clip_grad_norm_(self.net.parameters(),0.5)#更新を抑える
        self.optim.step()



    def predict(self,x):

        self.net.eval()

        with torch.no_grad():

            x=tensor(x,dtype=float)

            V,mu,var=self.net(x.float())

        return V.detach().numpy(),mu.detach().numpy(),var.detach().numpy()


    def save(self,name=str()):


        torch.save(self.net.state_dict(),name)



        return 0

    def load(self,name=str()):

        self.net.load_state_dict(torch.load(name))






def recommend(ranked_list=[]):

    zeus=A2C_zeus(model(),'zeus',2,0.4,0.99,'zeus')

    return zeus.recommend(ranked_list)

def learn_zeus(input_data=[]):

    zeus=A2C_zeus(model(),'zeus',2,0.4,0.99,'zeus')

    return zeus.learn(input_data)

def img_path2ranked_list(load_dir,save_dir):
    
    zeus=A2C_zeus(model(),'zeus',2,0.4,0.99,'zeus')
    
    return zeus.img_path2ranked_list(load_dir,save_dir)

def teach(n=int()):

    input_data=[]

    zeus=A2C_zeus(model(),'zeus',2,0.4,0.99,'zeus')

    for i in range(n):



        input_data.append(zeus.manual_teach())


    return zeus.learn(input_data)










