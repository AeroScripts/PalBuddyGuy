JawRight = 0 # +JawX
JawLeft = 1 # -JawX
JawForward = 2
JawOpen = 3
MouthApeShape = 4
MouthUpperRight = 5 # +MouthUpper
MouthUpperLeft = 6 # -MouthUpper
MouthLowerRight = 7 # +MouthLower
MouthLowerLeft = 8 # -MouthLower
MouthUpperOverturn = 9
MouthLowerOverturn = 10
MouthPout = 11
MouthSmileRight = 12 # +SmileSadRight
MouthSmileLeft = 13 # +SmileSadLeft
MouthSadRight = 14 # -SmileSadRight
MouthSadLeft = 15 # -SmileSadLeft
CheekPuffRight = 16
CheekPuffLeft = 17
CheekSuck = 18
MouthUpperUpRight = 19
MouthUpperUpLeft = 20
MouthLowerDownRight = 21
MouthLowerDownLeft = 22
MouthUpperInside = 23
MouthLowerInside = 24
MouthLowerOverlay = 25
TongueLongStep1 = 26
TongueLongStep2 = 32
TongueDown = 30 # -TongueY
TongueUp = 29 # +TongueY
TongueRight = 28 # +TongueX
TongueLeft = 27 # -TongueX
TongueRoll = 31
TongueUpLeftMorph = 34
TongueUpRightMorph = 33
TongueDownLeftMorph = 36
TongueDownRightMorph = 35


ENABLE_LOGGING = True
ENABLE_DISPLAY = True
DATASET_FOLDER = "A:\\Unity\\_Assets\\Special\\PalBuddyGuy\\sranidatisets\\"

max_power = 0.9 # the maximum threshold. Values will be adjusted to -1...1 range based on this

# dataset groups. .pkl files of the same paramter/class should be in the same list. Each primary key is a parameter. Names must match the name you specified when recording, plus "-em.pkl"
#order = [["neutral2-em.mmap", "neutral1-em.mmap", "nut3-em.mmap"], ["happy-em.mmap", "happy3-em.mmap"], ["mad1.mmap", "mad2.mmap", "mad4-em.mmap"], ["sad-em.mmap"], ["purseleft-em.mmap"], ["purseright-em.mmap"], ["open-em.mmap"], ["pog-em.mmap"], ["nwigleft-em.mmap"], ["nwigright-em.mmap"], ["showteth-em.mmap"]] # , ["weirdchamp-em.mmap"]

order = [["neutral2-em.mmap", "neutral1-em.mmap"], ["happy-em.mmap", "happy3-em.mmap"], ["mad-em.mmap"], ["sad-em.mmap"], ["open-em.mmap"], ["pog-em.mmap"], ["showteth-em.mmap"], ["upperupleft-em.mmap"]]

to_replace = {2: JawRight, 1: JawLeft, 4: MouthUpperUpLeft, 5: MouthUpperUpRight, 7: JawForward, 3: MouthPout}
to_replace_count = len(to_replace)


# can be auto calculated with fastcal command
max_power_array = [ max_power for e in range(to_replace_count) ]


import socket
import sys
import os

vrcft = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
vrcft.bind(("", 26421))
vrcft.listen(1)

print("Listening on ports 18452, 18455, 26421.")

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

highest_tid = 0

highest_hwid = 0
connection_count = 0

neural_queue = []

last_eye = None
swapped = False

target_stream = None # stream to write parameters to
target_stream_is_valid = False # if the stream is currently valid

def write_float(f):
    f = (f + 1) * 32767.0
    f = min(65535, f)
    f = max(0, f)
    f = int(f)
    target_stream.sendall(bytes([f // 256, f % 256]))
    
def write_param(key, value):
    target_stream.sendall(bytes([key,]))
    write_float(value)
    
def write_params(tensor):

    # normalize
    #tensor = torch.clip(((tensor / max_power_array) * 2) - 1, -1, 1)#torch.clip(tensor * 2 / max_power - 1, -1, 1)

    target_stream.sendall(bytes([2,]))
    target_stream.sendall(bytes([to_replace_count,]))
    i = 0
    for key in to_replace.keys():
        write_param(to_replace[key], torch.clip(((tensor[key] / max_power_array[i]) * 2) - 1, -1, 1))
        i = i + 1

def reader_thread():
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", 18454))
                
                def read(length):
                    v = s.recv(length)
                    while len(v) < length:
                        v = v + s.recv(length - len(v))
                    return v

                while True:
                    neural_queue.append((read(102400), read(102400)))
                    if len(neural_queue) > 60 * 4:
                        neural_queue.pop(0)
        except:
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

import torch
import pickle
import random
from tqdm import tqdm
DEVICE = torch.device('cuda') if torch.cuda.is_available() else "cpu:0"

batch_size = 128
epochs = 20

infer_output_paused = False
do_quick_cal = False
logger_string = "".join(["%.3f " for e in range(len(order))]).rstrip()

def neural_thread():
    global swapped
    global batch_size
    global do_quick_cal
    global max_power_array
    
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
                
            output_file = DATASET_FOLDER + str(dname) + "-em.pkl"
            with open(output_file, "wb") as w:
                pickle.dump(dataset_block, w)
                
        elif com == "save":
            torch.save({"conv1": conv1.state_dict(), "conv2": conv2.state_dict(), "linear1": linear1.state_dict(), "linear2": linear2.state_dict()}, "./buddyguy.pt")
            print("Saved model!")
        elif com == "load":
            load()
        elif com == "stats":
            start = time.time()
            tc = 0
            neural_queue.clear()
            while time.time() - start < 5:
                if len(neural_queue) > 0:
                    neural_queue.pop(0)
                    tc = tc + 1
                    
            print("%d fps" % (tc / 5))
        elif com == "convertmmap":
            print("Coverting checkpoints to mmaps...")
            file_list = os.listdir(DATASET_FOLDER)
            for file in tqdm(file_list):
                if file[-3:] == "pkl":
                    block = pickle.load(open(DATASET_FOLDER + file, "rb"))
                    print(len(block))
                    mmap_file = np.memmap(DATASET_FOLDER + file[:-3] + "mmap", dtype='float32', mode='w+', shape=(len(block), 128, 20, 20))
                    index = 0
                    for e in block:
                        a = torch.cat((torch.from_numpy(e[0]), torch.from_numpy(e[1])), dim=0).numpy()
                        mmap_file[index] = a
                        index = index + 1
                    mmap_file.flush()
                    del mmap_file
                    #[torch.cat((torch.from_numpy(e[0]), torch.from_numpy(e[1])), dim=0) for e in pickle.load(open(DATASET_FOLDER + set, "rb"))]
        elif com == "infer":
            with torch.no_grad():
            
                # ugly
                batch_size = 1
                def ugly(x):
                    return x
                dropout = ugly
                dropout2 = ugly
                
                #uglier
                def subcommands():
                    global infer_output_paused
                    global do_quick_cal
                    while True:
                        cmd = input()
                        infer_output_paused = True
                        print("\nPlease enter command...")
                        cmd = input()
                        if cmd == "fastcal":
                            print("Running fast calibration! Avatar will be puppeted, follow the puppet to calibrate thresholds.")
                            print("Starting in 10 seconds...")
                            time.sleep(10)
                            do_quick_cal = True
                            
                        infer_output_paused = False
                        
                threading.Thread(target=subcommands, args=()).start()
                
                if need_load:
                    load()
                    
                first = True
                
                def full_predict():
                    while len(neural_queue) == 0:
                        time.sleep(1)
                        print("Waiting for data...")
                    d = neural_queue[-1]
                    current_data = torch.cat((torch.from_numpy(decode_neural(d[0])), torch.from_numpy(decode_neural(d[1]))), dim=0).cuda().unsqueeze(0)
                    pred = predict(current_data)[0].cpu()
                    return pred
                
                full_predict()
                print("\n\nInfer started! Press enter to specify a command, like fastcal\n\n")
                
                while True:
                    if do_quick_cal:
                        # zero out params for puppeting
                        if target_stream_is_valid:
                            target_stream.sendall(bytes([2,]))
                            target_stream.sendall(bytes([to_replace_count,]))
                            for key in to_replace.keys():
                                write_param(to_replace[key], -1)
                        
                        quick_cal_result = []
                        e = 0
                        for key in to_replace.keys():
                        #for e in range(to_replace_count):
                            print("Puppeting %d" % e)
                            # ease-in puppet
                            if target_stream_is_valid:
                                for i in range(100):
                                    target_stream.sendall(bytes([2, 1, to_replace[key]]))
                                    write_float((i / 50) - 1)
                                    time.sleep(0.01)
                                target_stream.sendall(bytes([2, 1, to_replace[key]]))
                                write_float(1)
                            
                            # puppet for 3 seconds, take data from 2nd-3rd second
                            time.sleep(2)
                            neural_queue.clear()
                            start_time = time.time()
                            
                            avg = 0
                            cnt = 0
                            while time.time() - start_time < 1:
                                avg = avg + float(full_predict()[key])
                                cnt = cnt + 1
                            avg = avg / cnt
                            quick_cal_result.append(avg + 1e-9)
                            #time.sleep(1)
                            
                            # ease-out puppet
                            if target_stream_is_valid:
                                for i in range(100):
                                    target_stream.sendall(bytes([2, 1, to_replace[key]]))
                                    write_float(1 - ((i / 50) - 1))
                                    time.sleep(0.01)
                                target_stream.sendall(bytes([2, 1, to_replace[key]]))
                                write_float(-1)
                            
                            e = e + 1
                        do_quick_cal = False
                        print("Finished FastCal!\nResults:")
                        print(quick_cal_result)
                        max_power_array = quick_cal_result
                        print("\n\n")

                    pred = full_predict()
                    if target_stream_is_valid:
                        write_params(pred)

                    if not infer_output_paused:
                        print(logger_string % tuple(pred.cpu().numpy()), end=' \r')
                    time.sleep(0.01)
                        
                    
        elif com == "train":
            need_load = False
            print("Training...")
            
            datasets = {}
            
            setid = 0
            for sets in order:
                datasets[setid] = []
                for set in sets:
                    datasets[setid].append(np.memmap(DATASET_FOLDER + set, dtype='float32', mode='r', shape=(2048, 128, 20, 20)))

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
                        sample = datasets[id][random.randint(0,len(datasets[id])-1)][random.randint(0,2047)]
                        #print(sample.shape)
                        #sample = torch.from_numpy(sample).cuda()
                        
                        batch.append(sample)
                        batch_mask.append(mask)
                        
                    #batch = torch.stack(batch, dim=0) 
                    batch = torch.from_numpy(np.stack(batch, axis=0)).cuda()
                    batch_mask = torch.stack(batch_mask, dim=0)
                    
                    pred = predict(batch)
                    
                    loss = mse(pred, batch_mask) * 100
                    
                    
                    loss.backward()
                    p.set_description("Avg: %f Loss: %f         " % (avg/100, float(loss)/100))
                    opt.step()
                    tl = tl + float(loss)
                avg = float(tl/((2048*(setid+1)) // batch_size))
            
        else:
            print("Invalid command! valid: swap record")
    
def vrcft_thread():
    global target_stream
    global target_stream_is_valid
    while True:
        try:
            c, addr = vrcft.accept()
            target_stream = c
            target_stream_is_valid = True
            print("Received connection from VRCFT")
            def read_int():
                v = c.recv(1)
                while len(v) < 2:
                    v = v + c.recv(1)

                return int.from_bytes(v, "big")

            def read_byte():
                v = c.recv(1)
                while len(v) < 1:
                    v = v + c.recv(1)

                return int.from_bytes(v, "big")

            def read_float():
                a = read_int()
                a = float(a) / 32767.0 - 1.0
                return a

            while True:
                rid = read_byte()
                if rid == 1:
                    floats = [ read_float() for e in range(37) ]
                elif rid == 2:
                    floats = [ read_float() for e in range(60) ]
                else:
                    print("Invalid RID! " + str(rid))
            target_stream_is_valid = False
            
        except:
            target_stream_is_valid = False
            import traceback
            traceback.print_exc()
            time.sleep(1)

threading.Thread(target=reader_thread, args=()).start()
threading.Thread(target=neural_thread, args=()).start()
threading.Thread(target=vrcft_thread, args=()).start()
