from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
    
    def size(self):
        return self.buffer_size

    def counts(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class ReplayBufferLowLayer(ReplayBuffer):

    def __init__(self, buffer_size):
        super(ReplayBufferLowLayer,self).__init__(buffer_size)

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            batch_info = random.sample(self.buffer, self.num_experiences)
        else:
            batch_info = random.sample(self.buffer, batch_size)
        state_pri,idle_driver,action,goal,state_higher,next_state_higher,reward,done,T,next_T,order_num,next_goal=[],[],[],[],[],[],[],[],[],[],[],[]
        for e in batch_info:
            # print(e[0])
            state_pri.append(e[0])
            idle_driver.append(e[1])
            action.append(e[2])
            goal.append(e[3])
            state_higher.append(e[4])
            next_state_higher.append(e[5])
            reward.append(e[6])
            done.append(e[7])
            T.append(e[8])
            next_T.append(e[9])
            order_num.append(e[10]) 
            next_goal.append(e[11])
            


        return state_pri,idle_driver,action,goal,state_higher,next_state_higher,reward,done,T,next_T,order_num,next_goal

    def add(self,state_pri,idle_driver, action,
        goal,state_higher,next_state_higher,reward,done,T,next_T,order_num,next_goal):
        experience = [state_pri,idle_driver, action,
        goal,state_higher,next_state_higher,reward,done,T,next_T,order_num,next_goal]
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)



class ReplayBufferHighLayer(ReplayBuffer):

    def __init__(self, buffer_size):
        super(ReplayBufferHighLayer,self).__init__(buffer_size)

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            batch_info = random.sample(self.buffer, self.num_experiences)
        else:
            batch_info = random.sample(self.buffer, batch_size)
        state,reward,new_state,done,T,next_T,goal=[],[],[],[],[],[],[]
        for e in batch_info:
            # print(e[0])
            state.append(e[0])
            reward.append(e[1])
            new_state.append(e[2])
            done.append(e[3])
            T.append(e[4])
            next_T.append(e[5])
            goal.append(e[6])
        
        return state,reward,new_state,done,T,next_T,goal

    def add(self, state, reward, new_state,done,T,next_T,goal):
        experience = [state,reward,new_state,done,T,next_T,goal]
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)


    