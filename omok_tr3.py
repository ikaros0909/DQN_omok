from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import datetime


EPISODES = 10000001

#set "KERAS_BACKEND=tensorflow"
#export KERAS_BACKEND=tensorflow
#DQN 에이전트
class DQNAgent:
	def __init__(self, action_size, stone_name):		
		self.load_model = False
		# 상태와 행동의 크기 정의
		self.state_size = (15, 15, 17)
		self.action_size = action_size
		# DQN 하이퍼파라미터
		self.epsilon = 1.
		self.epsilon_start, self.epsilon_end = 1.0, 0.1
		self.exploration_steps = 1000000.
		self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
		self.batch_size = 32
		self.train_start = 50000
		self.update_target_rate = 1000
		self.discount_factor = 0.99
		# 리플레이 메모리, 최대 크기
		self.memory = deque(maxlen=50000)
		
		# 모델과 타겟모델을 생성하고 타겟모델 초기화
		self.model = self.build_model()
		self.target_model = self.build_model()
		self.update_target_model()

		self.optimizer = self.optimizer()
		
		if self.load_model and stone_name == "black":
			self.model.load_weights("./save_model/omok_black_v0.3.h5")
		elif self.load_model and stone_name == "white":
			self.model.load_weights("./save_model/omok_white_v0.3.h5")
			
    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
	def optimizer(self):
		a = K.placeholder(shape=(None,), dtype='int32')
		y = K.placeholder(shape=(None,), dtype='float32')

		prediction = self.model.output

		a_one_hot = K.one_hot(a, self.action_size)
		q_value = K.sum(prediction * a_one_hot, axis=1)
		error = K.abs(y - q_value)

		quadratic_part = K.clip(error, 0.0, 1.0)
		linear_part = error - quadratic_part
		loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

		optimizer = RMSprop(lr=0.00025, epsilon=0.01)
		updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
		train = K.function([self.model.input, a, y], [loss], updates=updates)

		return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
	def build_model(self):
		model = Sequential()		
		model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', input_shape=self.state_size))
		model.add(Conv2D(64, (5, 5), strides=(1, 1), activation='relu'))
		model.add(Conv2D(64, (5, 5), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dense(self.action_size))
		model.summary()
		return model

    # 타겟 모델을 모델의 가중치로 업데이트
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
	def get_action(self, history):
		
		if np.random.rand() <= self.epsilon:
			while True:
				_x = random.randrange(15)
				_y = random.randrange(15)
				if history[0,_x,_y,0] == 0 and history[0,_x,_y,8] == 0:
					return _x*15+_y
			
		else:
			q_value = self.model.predict(history)
			while True:
				_action = np.argmax(q_value[0])				
				if history[0,_action//15,_action%15,0] == 0 and history[0,_action//15,_action%15,8] == 0:
					return _action
				q_value[0,_action] = np.min(q_value[0])-1
    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
	def append_sample(self, history, action, reward, next_history):
		self.memory.append((history, action, reward, next_history))

	# 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
	def train_model(self):
		if self.epsilon > self.epsilon_end:
			self.epsilon -= self.epsilon_decay_step

		mini_batch = random.sample(self.memory, self.batch_size)

		history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
		next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
		target = np.zeros((self.batch_size,))
		action, reward = [], []

		for i in range(self.batch_size):
			history[i] = mini_batch[i][0]
			next_history[i] = mini_batch[i][3]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])			

		target_value = self.target_model.predict(next_history)

		for i in range(self.batch_size):
			target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

		loss = self.optimizer([history, action, target])		

def five_check(_x, _y, _board, _move):
	
	five_board = np.copy(_board)

	five = 1
	k = _x	
	while True:
		k -= 1
		if k < 0:
			break
		
		if five_board[k,_y] != 0 and five_board[k,_y]%2 == _move%2:
			five += 1
		else:
			break
	
	k = _x	
	while True:
		k += 1
		if k > 14:
			break
		
		if five_board[k,_y] != 0 and five_board[k,_y]%2 == _move%2:
			five += 1
		else:
			break

	if five ==5 :
		return True
		
	five = 1
	l = _y	
	while True:
		l -= 1
		if l < 0:
			break
		
		if five_board[_x,l] !=0 and five_board[_x,l]%2 == _move%2:
			five += 1
		else:
			break
	
	l = _y
	while True:
		l += 1
		if l > 14:
			break
		
		if five_board[_x,l] !=0 and five_board[_x,l]%2 == _move%2:
			five += 1
		else:
			break

	if five ==5 :
		return True
		
	five = 1
	k = _x
	l = _y
	
	while True:
		k -= 1
		l -= 1
		
		if k < 0 or l < 0 :
			break
		
		if five_board[k,l] !=0 and five_board[k,l]%2 == _move%2:
			five += 1
		else:
			break
			
	k = _x
	l = _y
	
	while True:
		k += 1
		l += 1
		
		if k > 14 or l > 14 :
			break
		
		if five_board[k,l] !=0 and five_board[k,l]%2 == _move % 2 :
			five += 1
		else:
			break		
	
	if five == 5 :
		return True
	
	five = 1
	k = _x
	l = _y
	
	while True:
		k += 1
		l -= 1
		
		if k > 14 or l < 0 :
			break
		
		if five_board[k,l] !=0 and five_board[k,l]%2 == _move % 2 :
			five += 1
		else:
			break
			
	k = _x
	l = _y
	
	while True:
		k -= 1
		l += 1
		
		if k < 0 or l > 14 :
			break
		
		if five_board[k,l] !=0 and five_board[k,l]%2 == _move % 2 :
			five += 1
		else:
			break		
	
	if five == 5 :
		return True
	

	return False
	

if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
	
	agent = [DQNAgent(action_size=225,stone_name="white"),DQNAgent(action_size=225,stone_name="black")]
	global_step = 0
	action = np.array([0,0])
	black = 0
	white = 0
	now = datetime.datetime.now()
	
	
	for e in range(EPISODES):
		done = False		
		op_done = False
		
		board = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]*225)
		board.shape = (15, 15, 17)
		board_no = np.array([0]*225)
		board_no.shape = (15,15)
		
		state = np.copy(board)
		state = np.reshape(state, [1, 15, 15, 17]) 
		
		for move in range(1, 226):
			#타켓모델을 일정 시간마다 업데이트 시, 필요한 변수
			global_step += 1
			
			#자신이 오목을 만들 수 있는 자리 찾기
			for i in range(225):
				if board_no[i//15,i%15] == 0:
					done = five_check(i//15, i%15, board_no, move)
					if done == True:
						action[move%2] = i
						break
			
			#상대가 오목을 만들 수 있는 자리 찾기
			if done == False:
				for i in range(225):
					if board_no[i//15,i%15] == 0:
						op_done = five_check(i//15, i%15, board_no, move+1)
						if op_done == True:
							action[move%2] = i							
							break
			
			#자신과 상대 모두 오목을 만들 수 없을 때
			if done == False and op_done == False:
				action[move%2] = agent[move%2].get_action(state)
			
			#액션 값을 x좌표와 y좌표로 분리
			move_x = action[move%2] // 15
			move_y = action[move%2] % 15
			
			#수순을 보기위한 배열, 액션 좌표에 숫자 입력
			board_no[move_x,move_y] = move
			
			#액션 값을 다음 착수상태 배열에 저장
			move_state = np.copy(state)
			move_state[:,:,:,1:7] = state[:,:,:,0:6]
			move_state[:,:,:,9:15] = state[:,:,:,8:14]
			move_state[:,:,:,16] = (move+1)%2
			move_state[0,move_x,move_y,(1-(move%2))*8] = 1
			
			if done == True:
				agent[(move)%2].append_sample(state, action[(move)%2], 1, move_state)
				agent[(move+1)%2].append_sample(undo_state, action[(move+1)%2], -1, move_state)
			elif move > 1:
				agent[(move+1)%2].append_sample(undo_state, action[(move+1)%2], 0, move_state)
				if move == 225:
						agent[(move)%2].append_sample(state, action[(move)%2], 0, move_state)
			
				
			if len(agent[(move)%2].memory) >= agent[(move)%2].train_start:						
				agent[(move)%2].train_model()
				
			if len(agent[(move+1)%2].memory) >= agent[(move+1)%2].train_start:						
				agent[(move+1)%2].train_model()
			
			# 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
			if global_step % agent[(move)%2].update_target_rate == 0:
				agent[(move)%2].update_target_model()
				agent[(move+1)%2].update_target_model()
				
			if done == True:
				if move%2 == 0:
					white += 1
				else:
					black += 1
				break
			elif move < 225:
				undo_state = np.copy(state)				
				state = np.copy(move_state)
			
		# 100 에피소드마다 모델 저장
		if e % 100 == 0:
			now1 = datetime.datetime.now()
			print(board_no)
			print("episode:", e, "  move:",move,"  black:",black,"  white:",white, "   time:",now1-now)			
			now = now1
			agent[1].model.save_weights("./save_model/omok_black_v0.3.h5")
			agent[0].model.save_weights("./save_model/omok_white_v0.3.h5")