
DATASET_FOLDER = "./"

# dataset groups. .pkl files of the same paramter/class should be in the same list. Each primary key is a parameter. Names must match the name you specified when recording, plus "-em.pkl"
order = [["neutral2-em.pkl", "neutral1-em.pkl"], ["happy-em.pkl"], ["mad1.pkl", "mad2.pkl"], ["sad-em.pkl"]]

import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 18452))

s.listen(1)

s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.bind(("", 18453))

s2.listen(1)

def decode_image(data, flipped=False):
    data = np.frombuffer(data, dtype=np.float32)
    data = np.clip(np.reshape(data, (1, 2, 100, 100)) * 255, 0, 255)[0]
    
    new_image = np.zeros((100, 200), dtype=np.float32)
    
    if flipped:
        new_image[:, :100] = np.flip(data[1], 1)
        new_image[:, 100:] = np.flip(data[0], 1)
    else:
        new_image[:, :100] = data[0]
        new_image[:, 100:] = data[1]
    
    new_image = np.stack((new_image, new_image, new_image), axis=-1).astype(np.uint8)
    return new_image

def decode_neural(data):
    data = np.frombuffer(data, dtype=np.float32)
    data = np.reshape(data, (1, 64, 20, 20))[0]
    return data

import time
import numpy as np
import cv2
import threading

lt = time.time()

backbuffer = np.zeros((200, 200, 3), dtype=np.uint8)
buffer = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.imshow("test", cv2.resize(buffer, (800, 800)))
cv2.waitKey(1)

highest_tid = 0

highest_hwid = 0
connection_count = 0

neural_queue = []

last_eye = None
swapped = False

def reader_thread(s, id):
    global highest_tid
    global highest_hwid
    global connection_count
    global lt
    global backbuffer
    global last_eye
    while True:
        try:
            c, addr = s.accept()
            print("Received connection")
            connection_count = connection_count + 1
            try:

                rid = 0
                while True:
                    def read(length):
                        v = c.recv(length)
                        while len(v) < length:
                            v = v + c.recv(length - len(v))
                        return v
                    
                    def read_int():
                        v = read(4)
                        return int.from_bytes(v, "little")

                    device = read_int() # broken
                    length = read_int()
                    thread_id = read_int()
                    
                    if id == 1 and length == 80000:
                        highest_hwid = (highest_hwid * 0.99) + (device * 0.01) # broken as fuck
                    
                    if thread_id > highest_tid:
                        highest_tid = thread_id
                    
                    if not (102400 == length or 80000 == length):
                        print("Invalid packet!")
                        try:
                            c.close()
                        except:
                            traceback.print_exc()
                        break

                    data = read(length)
                    
                    if length == 80000: # camera frame
                        if (id == 0 and (not swapped)) or (id == 1 and swapped):
                            buffer[100:200, :, :] = decode_image(data, flipped=True)
                            backbuffer = buffer.copy()
                        else:
                            buffer[:100, :, :] = decode_image(data)
                    else:
                        if (id == 0 and (not swapped)) or (id == 1 and swapped):
                            if last_eye is not None:
                                neural_queue.append((last_eye, data))
                                if len(neural_queue) > 60 * 4:
                                    del neural_queue[:len(neural_queue)-(60 * 4)]
                        else:
                            last_eye = data

                    rid = rid + 1
            except:
                import traceback
                traceback.print_exc()
                try:
                    c.close()
                except:
                    traceback.print_exc()
                time.sleep(0.1)
        except:
            import traceback
            traceback.print_exc()
        connection_count = connection_count - 1

import torch
import pickle
import random
from tqdm import tqdm
DEVICE = torch.device('cuda') if torch.cuda.is_available() else "cpu:0"

batch_size = 128
epochs = 20

def neural_thread():
    global swapped
    global batch_size
    print("Neural thread started!")
    
    conv1 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1).cuda()
    conv2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1).cuda()
    linear1 = torch.nn.Linear(25600, 1024).cuda()
    linear2 = torch.nn.Linear(1024, len(order)).cuda()

    act = torch.nn.ReLU()
    mse = torch.nn.MSELoss()
    dropout = torch.nn.Dropout(p=0.2)
    dropout2 = torch.nn.Dropout(p=0.7)
    
    opt = torch.optim.Adam(list(linear1.parameters()) + list(linear2.parameters()) + list(conv1.parameters())+ list(conv2.parameters()), lr=5e-5)

    need_load = True

    def load():
        dicts = torch.load("./buddyguy.pt")
        conv1.load_state_dict(dicts["conv1"])
        conv2.load_state_dict(dicts["conv2"])
        linear1.load_state_dict(dicts["linear1"])
        linear2.load_state_dict(dicts["linear2"])
        print("Loaded model!")
        
    def predict(batch):
        batch = act(conv1(dropout2(batch)))
        y = batch
        batch = act(conv2(dropout(batch)))
        batch += y
        pred = linear1(dropout(torch.reshape(batch, (batch_size, 25600))))
        pred = act(pred)
        pred = linear2(dropout(pred))
        pred = act(pred)
        return pred

    while True:
        print("Please enter cmd!")
        com = input()
        if com == "swap":
            print("inputs swapped!")
            swapped = not swapped
        elif com == "record":
            print("Please enter dataset name!")
            dname = input()
            print("Starting in 5 seconds...")
            time.sleep(5)
            dataset_block = []
            neural_queue.clear()
            while len(dataset_block) < 2048:
                while len(neural_queue) == 0:
                    time.sleep(0.001)
                d = neural_queue.pop(0)
                dataset_block.append((decode_neural(d[0]), decode_neural(d[1])))
                print("%d / 2048    " % len(dataset_block), end='\r')
                
            with open(DATASET_FOLDER + dname + "-em.pkl", "wb") as w:
                pickle.dump(dataset_block, w)
                
        elif com == "save":
            torch.save({"conv1": conv1.state_dict(), "conv2": conv2.state_dict(), "linear1": linear1.state_dict(), "linear2": linear2.state_dict()}, "./buddyguy.pt")
            print("Saved model!")
        elif com == "load":
            load()
        elif com == "infer":
            with torch.no_grad():
            
                # ugly
                batch_size = 1
                def ugly(x):
                    return x
                dropout = ugly
                dropout2 = ugly
                
                
                if need_load:
                    load()
                while True:
                    while len(neural_queue) == 0:
                        time.sleep(1)
                        print("Waiting for data...")
                    d = neural_queue[-1]
                    current_data = torch.cat((torch.from_numpy(decode_neural(d[0])), torch.from_numpy(decode_neural(d[1]))), dim=0).cuda().unsqueeze(0)
                    pred = predict(current_data)[0].cpu()
                    print(pred, end='        \r')
                    time.sleep(0.01)
        elif com == "train":
            need_load = False
            print("Training...")
            
            datasets = {}
            
            setid = 0
            for sets in order:
                datasets[setid] = []
                for set in sets:
                    print("Loading " + set)
                    datasets[setid] += [torch.cat((torch.from_numpy(e[0]), torch.from_numpy(e[1])), dim=0) for e in pickle.load(open(DATASET_FOLDER + set, "rb"))]
                    print(datasets[setid][0].shape)
                setid = setid + 1
            
            setid = setid - 1
            
            masks = {}
            for e in range(setid+1):
                masks[e] = np.zeros((setid+1, ), dtype=np.float32)
                masks[e][e] = 1
                masks[e] = torch.from_numpy(masks[e]).cuda()
            
            
            avg = 0
            ep = tqdm(range(epochs))
            for e in ep:
                p = tqdm(range((2048*(setid+1)) // batch_size), leave=False)
                tl = 0
                for s in p:
                    opt.zero_grad()
                    batch = []
                    batch_mask = []
                    for b in range(batch_size):
                        id = random.randint(0,setid)
                        mask = masks[id]
                        sample = datasets[id][random.randint(0,len(datasets[id])-1)].cuda()
                        
                        batch.append(sample)
                        batch_mask.append(mask)
                        
                    batch = torch.stack(batch, dim=0) 
                    batch_mask = torch.stack(batch_mask, dim=0)
                    
                    pred = predict(batch)
                    
                    loss = mse(pred, batch_mask)
                    
                    
                    loss.backward()
                    p.set_description("Avg: %f Loss: %f         " % (avg, float(loss)))
                    opt.step()
                    tl = tl + float(loss)
                avg = float(tl/((2048*(setid+1)) // batch_size))
            
        else:
            print("Invalid command! valid: swap record")
    


threading.Thread(target=reader_thread, args=(s, 0)).start()
threading.Thread(target=reader_thread, args=(s2, 1)).start()
threading.Thread(target=neural_thread, args=()).start()


while True:
    if time.time() - lt > 0.01:
        lt = time.time()
        cv2.imshow("test", cv2.resize(backbuffer, (800, 800)))
        cv2.waitKey(1)
